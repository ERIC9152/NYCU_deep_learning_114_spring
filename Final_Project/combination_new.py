#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
combination.py
把多個模型的機率輸出 (prob_fake) 合併成一份 ensemble 機率與最終標籤。

假設：
- prob_fake 越大越像 fake（你已確認）
- 每個輸入 CSV 至少包含兩欄：filename, prob_fake
  （如果還有 label 欄位也沒關係，會被忽略）

輸出：
1) out_prob_csv: filename, prob_fake  (ensemble 後的機率)
2) out_csv:      filename, label     (套 threshold 後的 real/fake)
3) 若使用 --scan，會額外輸出多個 submission_thr_*.csv 與 scan_summary.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_prob_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"filename", "prob_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} 欄位不足，需要至少包含 {sorted(list(need))}，但目前是 {list(df.columns)}")
    # 避免 filename 被讀成數字導致合併出事
    df["filename"] = df["filename"].astype(str)
    df["prob_fake"] = df["prob_fake"].astype(float)
    return df[["filename", "prob_fake"]]


def weighted_average(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights / (weights.sum() + 1e-12)
    return (probs * w[None, :]).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_csvs", type=str, nargs="+", required=True,
                    help="多個包含 (filename,prob_fake) 的 CSV 路徑")
    ap.add_argument("--weights", type=float, nargs="*", default=None,
                    help="（可選）每個模型的權重，數量需等於 prob_csvs")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="prob_fake >= threshold -> fake，否則 real")
    ap.add_argument("--out_prob_csv", type=str, default="ensemble_prob.csv",
                    help="輸出 ensemble 機率 CSV")
    ap.add_argument("--out_csv", type=str, default="ensemble_submission.csv",
                    help="輸出最終 submission CSV (filename,label)")
    ap.add_argument("--scan", action="store_true",
                    help="啟用 threshold 掃描，會在 out_dir 產多份 submission")
    ap.add_argument("--scan_from", type=float, default=0.35)
    ap.add_argument("--scan_to", type=float, default=0.65)
    ap.add_argument("--scan_step", type=float, default=0.01)
    ap.add_argument("--out_dir", type=str, default="thr_scan_outputs",
                    help="threshold 掃描輸出資料夾")
    args = ap.parse_args()

    paths = [Path(p) for p in args.prob_csvs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"找不到檔案：{p}")

    # 權重
    if args.weights is None or len(args.weights) == 0:
        weights = np.ones(len(paths), dtype=np.float32)
    else:
        if len(args.weights) != len(paths):
            raise ValueError("weights 數量必須等於 prob_csvs 數量")
        weights = np.array(args.weights, dtype=np.float32)

    # 讀檔 + merge
    merged = None
    for i, p in enumerate(paths):
        df = read_prob_csv(p).rename(columns={"prob_fake": f"p{i}"})
        merged = df if merged is None else merged.merge(df, on="filename", how="inner")

    if merged is None or len(merged) == 0:
        raise RuntimeError("合併後沒有資料（可能是 filename 不一致或檔案為空）")

    # ensemble prob
    prob_mat = merged[[f"p{i}" for i in range(len(paths))]].to_numpy(dtype=np.float32)
    ens_prob = weighted_average(prob_mat, weights)

    out_prob = pd.DataFrame({"filename": merged["filename"].values, "prob_fake": ens_prob})
    Path(args.out_prob_csv).parent.mkdir(parents=True, exist_ok=True)
    out_prob.to_csv(args.out_prob_csv, index=False)

    # single threshold submission
    labels = np.where(ens_prob >= float(args.threshold), "fake", "real")
    out_sub = pd.DataFrame({"filename": merged["filename"].values, "label": labels})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_sub.to_csv(args.out_csv, index=False)

    fake_ratio = float(np.mean(labels == "fake"))
    print(f"[OK] ensemble_prob -> {args.out_prob_csv}")
    print(f"[OK] submission(thr={args.threshold:.3f}) -> {args.out_csv} | fake_ratio={fake_ratio:.3f}")

    # threshold scan（沒有真實標籤時，只能產檔 + 觀察 fake_ratio，LB 你再上傳比）
    if args.scan:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        thr_list = np.arange(args.scan_from, args.scan_to + 1e-12, args.scan_step)
        for thr in thr_list:
            lab = np.where(ens_prob >= float(thr), "fake", "real")
            out_path = out_dir / f"submission_thr_{thr:.3f}.csv"
            pd.DataFrame({"filename": merged["filename"].values, "label": lab}).to_csv(out_path, index=False)
            rows.append({
                "threshold": float(thr),
                "fake_ratio": float(np.mean(lab == "fake")),
                "out_csv": str(out_path)
            })

        summary = pd.DataFrame(rows).sort_values("threshold")
        summary_path = out_dir / "scan_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[OK] scan outputs -> {out_dir}")
        print(f"[OK] scan summary  -> {summary_path}")


if __name__ == "__main__":
    main()
