import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_csv", type=str, required=True, help="(filename,prob_fake)")
    ap.add_argument("--out_dir", type=str, default="thr_scan_outputs")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.45,0.46,0.47,0.48,0.49,0.50])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.prob_csv)
    if "filename" not in df.columns or "prob_fake" not in df.columns:
        raise ValueError("prob_csv 需要包含欄位：filename, prob_fake")

    rows = []
    for thr in args.thresholds:
        out_path = out_dir / f"submission_thr_{thr:.3f}.csv"
        label = np.where(df["prob_fake"].values >= thr, "fake", "real")
        sub = pd.DataFrame({"filename": df["filename"].values, "label": label})
        sub.to_csv(out_path, index=False)

        fake_ratio = float(np.mean(label == "fake"))
        rows.append({"threshold": thr, "fake_ratio": fake_ratio, "out_csv": str(out_path)})

    summary = pd.DataFrame(rows).sort_values("threshold")
    summary_path = out_dir / "scan_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("[OK] Done. Summary:")
    print(summary)
    print(f"[OK] Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
