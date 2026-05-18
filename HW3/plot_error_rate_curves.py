
# plot_error_rate_curves.py
# Usage:
#   python plot_error_rate_curves.py --root runs --runs LSTM2_h224_L140_lr2p5e-3_clip075_ep15 RNN2_h224_L140_lr2p5e-3_clip075_ep15
#   python plot_error_rate_curves.py --root runs --all --model lstm
#
import argparse, csv, os
from pathlib import Path
import matplotlib.pyplot as plt

BREAKS = [3, 6, 9, 12, 15]

def find_acc_keys(header):
    # Try common Keras CSVLogger headers
    cands = ["accuracy", "sparse_categorical_accuracy", "acc"]
    for k in cands:
        if k in header:
            train = k
            break
    else:
        train = None
    for k in ["val_"+c for c in cands]:
        if k in header:
            val = k
            break
    else:
        val = None
    return train, val

def read_history_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames or []
        tkey, vkey = find_acc_keys(header)
        for i, r in enumerate(rd):
            row = {"epoch": int(r.get("epoch", i))}
            # loss
            try:
                row["loss"] = float(r.get("loss")) if r.get("loss") is not None else None
            except:
                row["loss"] = None
            try:
                row["val_loss"] = float(r.get("val_loss")) if r.get("val_loss") is not None else None
            except:
                row["val_loss"] = None
            # accuracy
            if tkey and r.get(tkey) not in (None, ""):
                try:
                    row["acc"] = float(r.get(tkey))
                except:
                    row["acc"] = None
            else:
                row["acc"] = None
            if vkey and r.get(vkey) not in (None, ""):
                try:
                    row["val_acc"] = float(r.get(vkey))
                except:
                    row["val_acc"] = None
            else:
                row["val_acc"] = None
            rows.append(row)
    return rows

def plot_one(run_dir, save_dir):
    run_dir = Path(run_dir)
    hist = run_dir / "history.csv"
    if not hist.exists():
        print(f"[skip] {run_dir.name}: no history.csv")
        return None
    rows = read_history_csv(hist)
    if not rows:
        print(f"[skip] {run_dir.name}: empty history.csv")
        return None
    epochs = [r["epoch"]+1 for r in rows]
    train_acc = [r["acc"] for r in rows]
    val_acc   = [r["val_acc"] for r in rows]
    train_err = [1.0 - a if a is not None else None for a in train_acc]
    val_err   = [1.0 - a if a is not None else None for a in val_acc]

    plt.figure(figsize=(8,5))
    # Error rate curves
    if any(e is not None for e in train_err):
        plt.plot(epochs, train_err, label="Train Error Rate", linewidth=2)
    if any(e is not None for e in val_err):
        plt.plot(epochs, val_err, label="Validation Error Rate", linewidth=2)
    # Breaking points
    ymax = max([y for y in (train_err+val_err) if y is not None] + [1.0])
    for bx in BREAKS:
        if bx <= epochs[-1]:
            plt.axvline(bx, color="gray", linestyle="--", linewidth=1, alpha=0.4)
            plt.text(bx, ymax*0.98, f"{bx}", ha="center", va="top", fontsize=8, color="gray")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (1 - Accuracy)")
    plt.title(f"{run_dir.name} | Error Rate vs Epoch")
    plt.grid(alpha=0.2)
    plt.legend()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{run_dir.name}_error_rate.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved:", out)
    return out

def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--runs", nargs="*", help="specific run folder names under --root")
    ap.add_argument("--all", action="store_true", help="plot for all runs under --root")
    ap.add_argument("--model", choices=["rnn","lstm"], help="filter by model name in config.json")
    ap.add_argument("--out", default="figs_error")
    args = ap.parse_args()

    root = Path(args.root)
    targets = []
    if args.all:
        for d in sorted(root.iterdir()):
            if not d.is_dir(): continue
            if args.model:
                cfg = d / "config.json"
                if cfg.exists():
                    try:
                        j = json.loads(cfg.read_text(encoding="utf-8"))
                        if j.get("model") != args.model:
                            continue
                    except:
                        pass
                else:
                    if args.model not in d.name.lower():
                        continue
            targets.append(d)
    else:
        if not args.runs:
            print("No runs specified. Use --all or --runs <names>")
            return
        for name in args.runs:
            d = root / name
            targets.append(d)

    out_paths = []
    for t in targets:
        p = plot_one(t, Path(args.out))
        if p: out_paths.append(str(p))

    if not out_paths:
        print("No plots generated. Check --root/--runs/--model settings.")

if __name__ == "__main__":
    main()
