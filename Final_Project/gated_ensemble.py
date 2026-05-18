#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read_prob(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"filename", "prob_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} needs columns {need}, got {df.columns.tolist()}")
    df["filename"] = df["filename"].astype(str)
    df["prob_fake"] = df["prob_fake"].astype(float)
    return df[["filename", "prob_fake"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_prob", required=True, help="主力模型 prob csv (filename,prob_fake)")
    ap.add_argument("--alt_prob", required=True, help="輔助模型 prob csv (filename,prob_fake)，例如 kfold3_auto_fine_prob.csv")
    ap.add_argument("--lo", type=float, default=0.39, help="模糊區下界")
    ap.add_argument("--hi", type=float, default=0.43, help="模糊區上界")
    ap.add_argument("--mode", choices=["replace","blend"], default="replace",
                    help="replace: 模糊區直接用 alt；blend: 模糊區用 alpha 加權混合")
    ap.add_argument("--alpha", type=float, default=0.5, help="blend 模式下：final=alpha*base+(1-alpha)*alt")
    ap.add_argument("--threshold", type=float, default=0.41, help="最終切 label 的 threshold")
    ap.add_argument("--out_prefix", default="gated", help="輸出檔名前綴")
    args = ap.parse_args()

    base = read_prob(args.base_prob).rename(columns={"prob_fake":"p_base"})
    alt  = read_prob(args.alt_prob ).rename(columns={"prob_fake":"p_alt"})
    m = base.merge(alt, on="filename", how="inner")
    if len(m)==0:
        raise RuntimeError("merge 後沒有資料，檢查 filename 是否一致")

    pb = m["p_base"].to_numpy()
    pa = m["p_alt"].to_numpy()

    mask = (pb >= args.lo) & (pb <= args.hi)
    if args.mode == "replace":
        pf = np.where(mask, pa, pb)
    else:
        pf = pb.copy()
        pf[mask] = args.alpha * pb[mask] + (1-args.alpha) * pa[mask]

    # 輸出 prob
    prob_out = pd.DataFrame({"filename": m["filename"].values, "prob_fake": pf})
    prob_path = f"{args.out_prefix}_prob.csv"
    prob_out.to_csv(prob_path, index=False)

    # 輸出 submission
    labels = np.where(pf >= args.threshold, "fake", "real")
    sub_out = pd.DataFrame({"filename": m["filename"].values, "label": labels})
    sub_path = f"{args.out_prefix}_sub.csv"
    sub_out.to_csv(sub_path, index=False)

    print(f"[OK] {prob_path}")
    print(f"[OK] {sub_path} | fake_ratio={(labels=='fake').mean():.4f} | gated_frac={mask.mean():.4f}")

if __name__ == "__main__":
    main()
