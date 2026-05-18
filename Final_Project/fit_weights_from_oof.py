#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"filename", "y_true", "prob_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 欄位不足，需要 {need}，目前 {df.columns.tolist()}")
    df["filename"] = df["filename"].astype(str)
    df["y_true"] = df["y_true"].astype(int)
    df["prob_fake"] = df["prob_fake"].astype(float)
    return df[["filename", "y_true", "prob_fake"]]

def read_testprob(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"filename", "prob_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 欄位不足，需要 {need}，目前 {df.columns.tolist()}")
    df["filename"] = df["filename"].astype(str)
    df["prob_fake"] = df["prob_fake"].astype(float)
    return df[["filename", "prob_fake"]]

def best_threshold_acc(y: np.ndarray, p: np.ndarray, step: float = 0.001):
    best_thr, best_acc = 0.5, -1.0
    thrs = np.arange(0.0, 1.0 + 1e-12, step)
    for t in thrs:
        pred = (p >= t).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)
    return best_thr, float(best_acc)

def grid_weights_3(step: float = 0.05):
    # w1+w2+w3=1, wi>=0
    vals = np.arange(0.0, 1.0 + 1e-12, step)
    W = []
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if w3 < 0:
                w3 = 0.0
            if abs(w1 + w2 + w3 - 1.0) < 1e-6:
                W.append((w1, w2, w3))
    return W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_csvs", nargs=3, required=True,
                    help="三個 OOF csv（含 filename,y_true,prob_fake）")
    ap.add_argument("--test_prob_csvs", nargs=3, required=True,
                    help="三個 test prob csv（含 filename,prob_fake）")
    ap.add_argument("--weight_step", type=float, default=0.05,
                    help="權重格點步長（越小越精，但越慢）")
    ap.add_argument("--thr_step", type=float, default=0.001,
                    help="threshold 掃描步長")
    ap.add_argument("--out_prefix", type=str, default="stacked",
                    help="輸出檔名前綴")
    args = ap.parse_args()

    oof_dfs = [read_oof(Path(p)) for p in args.oof_csvs]
    # 以 filename inner join 對齊（確保順序一致）
    merged = oof_dfs[0].rename(columns={"prob_fake":"p0"})
    merged = merged.merge(oof_dfs[1].rename(columns={"prob_fake":"p1"}), on=["filename","y_true"], how="inner")
    merged = merged.merge(oof_dfs[2].rename(columns={"prob_fake":"p2"}), on=["filename","y_true"], how="inner")

    y = merged["y_true"].to_numpy(dtype=int)
    P = merged[["p0","p1","p2"]].to_numpy(dtype=np.float32)

    best = {"acc": -1.0, "thr": 0.5, "w": (1/3,1/3,1/3)}

    # 先粗搜權重，再對每個權重找最佳 threshold（accuracy）
    weights = grid_weights_3(step=args.weight_step)
    for w in weights:
        wv = np.array(w, dtype=np.float32)
        p_ens = (P * wv[None, :]).sum(axis=1)
        thr, acc = best_threshold_acc(y, p_ens, step=args.thr_step)
        if acc > best["acc"]:
            best = {"acc": acc, "thr": thr, "w": w}

    print("\n[BEST on OOF]")
    print(f"  weights (p0,p1,p2) = {best['w']}")
    print(f"  best_thr = {best['thr']:.4f}")
    print(f"  oof_acc  = {best['acc']:.6f}")

    # 讀 test prob 並對齊
    test_dfs = [read_testprob(Path(p)) for p in args.test_prob_csvs]
    testm = test_dfs[0].rename(columns={"prob_fake":"p0"})
    testm = testm.merge(test_dfs[1].rename(columns={"prob_fake":"p1"}), on="filename", how="inner")
    testm = testm.merge(test_dfs[2].rename(columns={"prob_fake":"p2"}), on="filename", how="inner")

    Pt = testm[["p0","p1","p2"]].to_numpy(dtype=np.float32)
    wv = np.array(best["w"], dtype=np.float32)
    ptest = (Pt * wv[None, :]).sum(axis=1)

    # 輸出 ensemble prob
    prob_out = pd.DataFrame({"filename": testm["filename"].values, "prob_fake": ptest})
    prob_path = f"{args.out_prefix}_prob.csv"
    prob_out.to_csv(prob_path, index=False)

    # 輸出 submission
    labels = np.where(ptest >= float(best["thr"]), "fake", "real")
    sub_out = pd.DataFrame({"filename": testm["filename"].values, "label": labels})
    sub_path = f"{args.out_prefix}_submission.csv"
    sub_out.to_csv(sub_path, index=False)

    fake_ratio = float(np.mean(labels=="fake"))
    print(f"\n[OK] saved -> {prob_path}")
    print(f"[OK] saved -> {sub_path} | fake_ratio={fake_ratio:.4f}")

if __name__ == "__main__":
    main()
