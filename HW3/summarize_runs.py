#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, json, math, argparse, re
from datetime import datetime
import pandas as pd

def natural_to_bpc(x):
    try:
        return float(x) / math.log(2.0)
    except Exception:
        return None

def read_bpc_log(path):
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        best = None; best_epoch=None; best_train_bpc=None
        for rec in data:
            vb = rec.get("val_bpc", None)
            if vb is None: 
                continue
            if (best is None) or (vb < best):
                best = vb; best_epoch = rec.get("epoch", None); best_train_bpc = rec.get("bpc", None)
        final_vb = data[-1].get("val_bpc", None) if data else None
        final_epoch = data[-1].get("epoch", None) if data else None
        return best, best_epoch, final_vb, final_epoch, best_train_bpc
    except Exception:
        return None, None, None, None, None

def read_history_csv(path):
    try:
        import csv
        rows = []
        with io.open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if not rows:
            return None, None, None, None, None
        best = None; best_epoch=None; best_train_bpc=None
        for idx, r in enumerate(rows, start=1):
            epoch = int(r.get("epoch", idx)) if "epoch" in r else idx
            val_loss = r.get("val_loss", None)
            loss = r.get("loss", None)
            vb = natural_to_bpc(val_loss) if val_loss not in (None,"") else None
            tb = natural_to_bpc(loss) if loss not in (None,"") else None
            if vb is not None and ((best is None) or (vb < best)):
                best = vb; best_epoch = epoch; best_train_bpc = tb
        last = rows[-1]
        final_vb = natural_to_bpc(last.get("val_loss", None))
        final_epoch = int(last.get("epoch", len(rows))) if "epoch" in last else len(rows)
        return best, best_epoch, final_vb, final_epoch, best_train_bpc
    except Exception:
        return None, None, None, None, None

def read_config_json(path):
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def guess_from_dirname(name):
    out = {}
    patterns = {
        "model": r"(lstm|gru|rnn)",
        "hidden": r"h(\d+)",
        "seq_len": r"L(\d+)",
        "lr": r"lr([0-9\.eE\-]+)",
        "batch": r"b(\d+)",
        "epochs": r"ep(\d+)",
        "seed": r"seed(\d+)",
    }
    import re
    for k, pat in patterns.items():
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            v = m.group(1)
            if k in ("hidden", "seq_len", "batch", "epochs", "seed"):
                try: out[k] = int(v)
                except: pass
            elif k == "lr":
                try: out[k] = float(v)
                except: pass
            else:
                out[k] = v.lower()
    return out

def find_weight_file(run_dir):
    for root, _, files in os.walk(run_dir):
        for fn in files:
            if fn.endswith(".weights.h5"):
                return os.path.join(root, fn)
    for root, _, files in os.walk(run_dir):
        for fn in files:
            if fn.endswith(".keras"):
                return os.path.join(root, fn)
    return None

def collect_one(run_dir):
    name = os.path.basename(run_dir.rstrip("/\\"))
    bpc_log = os.path.join(run_dir, "bpc_log.json")
    hist_csv = os.path.join(run_dir, "history.csv")
    cfg_json = os.path.join(run_dir, "config.json")

    best, best_ep, final_vb, final_ep, best_train = (None,)*5
    if os.path.exists(bpc_log):
        best, best_ep, final_vb, final_ep, best_train = read_bpc_log(bpc_log)
    if best is None and os.path.exists(hist_csv):
        best, best_ep, final_vb, final_ep, best_train = read_history_csv(hist_csv)

    cfg = read_config_json(cfg_json)
    guessed = guess_from_dirname(name)
    for k,v in guessed.items():
        cfg.setdefault(k, v)

    wfile = find_weight_file(run_dir)

    try:
        mtime = max(os.path.getmtime(os.path.join(run_dir,f)) for f in os.listdir(run_dir))
    except Exception:
        mtime = None

    return {
        "run_name": name,
        "path": os.path.abspath(run_dir),
        "model": cfg.get("model"),
        "hidden": cfg.get("hidden"),
        "seq_len": cfg.get("seq_len"),
        "batch": cfg.get("batch"),
        "epochs": cfg.get("epochs"),
        "lr": cfg.get("lr"),
        "seed": cfg.get("seed"),
        "best_val_bpc": best,
        "best_epoch": best_ep,
        "final_val_bpc": final_vb,
        "final_epoch": final_ep,
        "train_bpc_at_best": best_train,
        "weights_or_model": wfile,
        "modified_time": (None if mtime is None else __import__("datetime").datetime.fromtimestamp(mtime).isoformat(timespec="seconds"))
    }

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Summarize Keras runs into a table")
    ap.add_argument("--root", default="runs", help="根資料夾（每個子資料夾是一個 run）")
    ap.add_argument("--save-name", default="summary", help="輸出檔名前綴（不含副檔名）")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[WARN] root={root} 不存在或不是資料夾")
        return

    rows = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        rows.append(collect_one(run_dir))

    if not rows:
        print("[INFO] 沒找到任何 run 資料夾。")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["best_val_bpc","final_val_bpc"], ascending=[True, True], na_position="last")

    csv_path = os.path.join(root, f"{args.save_name}.csv")
    md_path  = os.path.join(root, f"{args.save_name}.md")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    cols = ["run_name","model","hidden","seq_len","batch","lr","epochs","best_val_bpc","best_epoch","final_val_bpc","weights_or_model"]
    md_df = df[[c for c in cols if c in df.columns]]
    try:
        with io.open(md_path, "w", encoding="utf-8") as f:
            f.write(md_df.to_markdown(index=False))
    except Exception:
        with io.open(md_path, "w", encoding="utf-8") as f:
            f.write(md_df.to_string(index=False))

    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] MD : {md_path}")

if __name__ == "__main__":
    main()
