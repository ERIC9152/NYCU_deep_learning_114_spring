#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def read_oof(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"filename", "y_true", "prob_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"[OOF] {path} needs columns {need}, got {df.columns.tolist()}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["y_true"] = df["y_true"].astype(int)
    df["prob_fake"] = df["prob_fake"].astype(float)
    return df[["filename", "y_true", "prob_fake"]]

def read_testprob(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"filename", "prob_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"[TEST] {path} needs columns {need}, got {df.columns.tolist()}")
    df = df.copy()
    df["filename"] = df["filename"].astype(str)
    df["prob_fake"] = df["prob_fake"].astype(float)
    return df[["filename", "prob_fake"]]

def merge_oof(oof_csvs):
    dfs = [read_oof(p).rename(columns={"prob_fake": f"p{i}"}) for i, p in enumerate(oof_csvs)]
    m = dfs[0]
    for i in range(1, len(dfs)):
        m = m.merge(dfs[i], on=["filename", "y_true"], how="inner")
    if len(m) == 0:
        raise RuntimeError("OOF merge 後資料為空：請檢查三份 oof 的 filename 是否一致")
    return m

def merge_test(test_csvs):
    dfs = [read_testprob(p).rename(columns={"prob_fake": f"p{i}"}) for i, p in enumerate(test_csvs)]
    m = dfs[0]
    for i in range(1, len(dfs)):
        m = m.merge(dfs[i], on=["filename"], how="inner")
    if len(m) == 0:
        raise RuntimeError("TEST merge 後資料為空：請檢查三份 test prob 的 filename 是否一致")
    return m

def best_threshold_acc(y, p, step=0.001):
    best_thr, best_acc = 0.5, -1.0
    thrs = np.arange(0.0, 1.0 + 1e-12, step)
    for t in thrs:
        pred = (p >= t).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)
    return best_thr, float(best_acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_csvs", nargs="+", required=True,
                    help="多個 OOF csv（需含 filename,y_true,prob_fake）")
    ap.add_argument("--test_prob_csvs", nargs="+", required=True,
                    help="多個 test prob csv（需含 filename,prob_fake），順序需與 oof 對應")
    ap.add_argument("--meta_folds", type=int, default=5,
                    help="meta 層交叉驗證 folds（建議 5）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0,
                    help="LogReg 正則強度（越大越弱正則）")
    ap.add_argument("--thr_step", type=float, default=0.001,
                    help="找最佳 threshold 的步長")
    ap.add_argument("--out_prefix", type=str, default="stacker",
                    help="輸出檔名前綴")
    args = ap.parse_args()

    if len(args.oof_csvs) != len(args.test_prob_csvs):
        raise ValueError("oof_csvs 與 test_prob_csvs 數量必須一致（順序也要對齊）")

    oofm = merge_oof(args.oof_csvs)
    testm = merge_test(args.test_prob_csvs)

    feat_cols = [c for c in oofm.columns if c.startswith("p")]
    X = oofm[feat_cols].to_numpy(dtype=np.float32)
    y = oofm["y_true"].to_numpy(dtype=int)

    Xt = testm[feat_cols].to_numpy(dtype=np.float32)

    # meta model：標準化 + logistic regression
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=args.C,
            max_iter=5000,
            solver="lbfgs"
        ))
    ])

    # 產 meta-oof（避免在同一份 OOF 上直接 fit+eval 造成樂觀偏差）
    skf = StratifiedKFold(n_splits=args.meta_folds, shuffle=True, random_state=args.seed)
    meta_oof = np.zeros(len(y), dtype=np.float32)

    for f, (tr, va) in enumerate(skf.split(X, y)):
        m = base_model
        m.fit(X[tr], y[tr])
        meta_oof[va] = m.predict_proba(X[va])[:, 1].astype(np.float32)
        print(f"[meta fold {f}] done")

    best_thr, best_acc = best_threshold_acc(y, meta_oof, step=args.thr_step)
    print("\n[STACKER meta-OoF]")
    print("  features =", feat_cols)
    print(f"  meta_folds = {args.meta_folds}")
    print(f"  best_thr = {best_thr:.4f}")
    print(f"  meta_oof_acc = {best_acc:.6f}")

    # 用全資料 fit 最終 stacker，再推 test
    base_model.fit(X, y)
    test_prob = base_model.predict_proba(Xt)[:, 1].astype(np.float32)

    # 輸出 meta_oof
    oof_out = pd.DataFrame({
        "filename": oofm["filename"].values,
        "y_true": y,
        "prob_fake": meta_oof
    })
    oof_path = f"{args.out_prefix}_meta_oof.csv"
    oof_out.to_csv(oof_path, index=False)

    # 輸出 test prob（給 gated_ensemble 用）
    test_out = pd.DataFrame({
        "filename": testm["filename"].values,
        "prob_fake": test_prob
    })
    test_path = f"{args.out_prefix}_test_prob.csv"
    test_out.to_csv(test_path, index=False)

    # 也順便輸出一份「用 meta-oof 最佳 thr」的 submission（可直接上 Kaggle 測）
    labels = np.where(test_prob >= best_thr, "fake", "real")
    sub_out = pd.DataFrame({"filename": testm["filename"].values, "label": labels})
    sub_path = f"{args.out_prefix}_bestthr_submission.csv"
    sub_out.to_csv(sub_path, index=False)

    print(f"\n[OK] {oof_path}")
    print(f"[OK] {test_path}")
    print(f"[OK] {sub_path} | fake_ratio={(labels=='fake').mean():.4f}")

if __name__ == "__main__":
    main()
