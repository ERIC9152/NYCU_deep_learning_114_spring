# -*- coding: utf-8 -*-
import argparse, json, math, os, csv
from pathlib import Path
import matplotlib.pyplot as plt

def read_config(run):
    p = run / "config.json"
    if not p.exists(): return None
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        def _to_int(x, default=None):
            try:
                return int(str(x).strip())
            except:
                return default
        return {
            "model":  cfg.get("model"),
            "hidden": _to_int(cfg.get("hidden")),
            "seq_len": _to_int(cfg.get("seq_len")),
            "batch":  _to_int(cfg.get("batch", 64), 64),
            "lr":     float(cfg.get("lr", 0.001)),
            "epochs": _to_int(cfg.get("epochs", 15), 15),
            "save_dir": str(cfg.get("save_dir", run.name)),
            "layers": _to_int(cfg.get("layers"), None),   # ← 加這行
        }
    except Exception:
        return None

def read_history(run):
    p = run / "history.csv"
    if not p.exists(): return None
    rows = []
    with p.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            # Keras CSVLogger 欄位：epoch,loss,accuracy,val_loss,val_accuracy...（你的code用 SparseCategoricalAccuracy）
            row = {k.strip(): r[k] for k in r}
            try:
                row["epoch"] = int(row.get("epoch", len(rows)))
            except:
                row["epoch"] = len(rows)
            for k in ["loss","val_loss"]:
                try:
                    row[k] = float(row[k])
                except:
                    row[k] = None
            rows.append(row)
    return rows

import re, sys
from pathlib import Path

def _infer_layers_from_name(run_name, default=1):
    m = re.search(r'(rnn|lstm|gru)\s*([0-9]+)', run_name, re.IGNORECASE)
    if m:
        try:
            return int(m.group(2))
        except:
            return default
    return default

def _parse_int_safe(v, default=None):
    try:
        if v is None: return default
        s = str(v).strip()
        if s == "": return default
        return int(s)
    except:
        return default

def collect_points(root, model_filter=None, which="train", stat="final",
                   min_layers=None, verbose=True):
    """
    回傳 points: (hidden, seq_len, bpc, model, run_name, layers)
    - 優先用 which 指定的 loss（train/val）；val 取不到就回退 train
    - layers 優先讀 config.json；若缺/空則從資料夾名推 (e.g., rnn2_*)
    - 可用 min_layers 篩掉層數不足的實驗
    """
    points = []
    root = Path(root)
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        cfg = read_config(run)
        if not cfg:
            if verbose: print(f"[skip] {run.name}: no config.json", file=sys.stderr)
            continue
        if model_filter and cfg["model"] != model_filter:
            continue

        hist = read_history(run)
        if not hist:
            if verbose: print(f"[skip] {run.name}: no history.csv", file=sys.stderr)
            continue

        # 先用指定 metric，取不到 val 就回退 train
        loss = pick_metric(hist, which=which, stat=stat)
        picked = which
        if loss is None and which == "val":
            loss = pick_metric(hist, which="train", stat=stat)
            picked = "train"
        if loss is None:
            if verbose: print(f"[skip] {run.name}: no usable loss (train/val)", file=sys.stderr)
            continue

        bpc = to_bpc(loss)

        # 取得 layers：config.json 優先，沒有就從資料夾名推
        layers = cfg.get("layers")
        if layers is None:
            layers = _infer_layers_from_name(run.name, 1)

        if (min_layers is not None) and (layers < min_layers):
            if verbose: print(f"[skip] {run.name}: layers={layers} < min_layers={min_layers}", file=sys.stderr)
            continue

        points.append((cfg["hidden"], cfg["seq_len"], bpc, cfg["model"], run.name, layers))

    if verbose:
        print(f"[info] collected {len(points)} points (model={model_filter or 'all'}, metric={picked}, min_layers={min_layers})", file=sys.stderr)
    return points

def to_bpc(x):  # cross-entropy (nats) -> bits/char
    return x / math.log(2.0)

def pick_metric(rows, which="train", stat="final"):
    """which: train/val ; stat: final/best"""
    key = "loss" if which=="train" else "val_loss"
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals: return None
    if stat == "best":
        return min(vals)  # 最小 loss
    else:
        return vals[-1]   # 最後一個 epoch


def plot(points, title, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import colors
    if not points:
        print("沒有找到可畫的點，請確認 runs/ 內是否含有 config.json 與 history.csv")
        return

    xs  = [p[0] for p in points]  # hidden
    ys  = [p[1] for p in points]  # seq_len
    cs  = np.array([p[2] for p in points], dtype=float)  # BPC (color)
    mods= [p[3] for p in points]
    lays= [p[5] for p in points]

    layer_markers = {1:'o', 2:'s', 3:'^', 4:'D'}
    uniq_layers = sorted(set(lays))
    uniq_models = sorted(set(mods))
    model_edge = {'lstm': 'black', 'rnn': 'dimgray', 'gru': 'tab:purple'}

    plt.figure(figsize=(8,5))

    # === 新增：固定 colormap 與縮放 ===
    cmap = plt.get_cmap("plasma")  # 對比高、連續
    # 用分位數避免極端值吃掉動態範圍；想用 min/max 就改成 cs.min(), cs.max()
    vmin, vmax = np.percentile(cs, [5, 95])
    if vmin == vmax:  # 極端情況保底
        vmin, vmax = cs.min(), cs.max() + 1e-6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # 先畫隱形散點來建立 colorbar（共用 cmap/norm）
    sc = plt.scatter(xs, ys, c=cs, s=180, cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, label="BPC (bits per character) ")

    # 真正逐點繪製（共用同一個 cmap/norm）
    for (x, y, cval, m, rn, L) in points:
        mk = layer_markers.get(L, 'o')
        ec = model_edge.get(m, 'black')
        plt.scatter(x, y, c=[cval], s=180, marker=mk,
                    edgecolors=ec, linewidths=1.2,
                    cmap=cmap, norm=norm)

    # 後面標題、圖例、儲存的程式碼維持不變…

    # 標題與軸名
    # 在 plot(...) 裡，計算完 uniq_models, uniq_layers 後加入：
    layer_text = ", ".join(str(L) for L in sorted(set(lays)))
    if len(uniq_models) == 1:
        subtitle = f"Layers shown: {layer_text}"
    else:
        subtitle = f"Models: {'/'.join(sorted(m.upper() for m in uniq_models))} | Layers shown: {layer_text}"

    plt.suptitle(f"{title}\n{subtitle}", y=0.98)   # 只用 suptitle，放主標＋副標
    # 刪除以下兩行，避免重覆與加字：
    # model_txt = "/".join(u.upper() for u in uniq_models)
    # plt.title(f"{title}\nModels: {model_txt} | Layers shown: {layer_text}")

    plt.xlabel("Hidden Size")
    plt.ylabel("Sequence Length")
    plt.tight_layout(rect=[0, 0, 1, 0.94])        # 預留上緣給 suptitle
    plt.grid(alpha=0.2)

    # ---- 圖例：層數（形狀）----
    handles_layers = [Line2D([0],[0], marker=layer_markers.get(L,'o'), color='w',
                             markerfacecolor='gray', markeredgecolor='gray',
                             markersize=9, label=f"L={L}") for L in uniq_layers]
    # ---- 圖例：模型（邊框色）----
    handles_models = [Line2D([0],[0], marker='o', color='w',
                             markerfacecolor='lightgray', markeredgecolor=model_edge.get(m,'black'),
                             markeredgewidth=1.5, markersize=9, label=m.upper()) for m in uniq_models]

    leg1 = plt.legend(handles=handles_layers, title="Layers (marker)", loc="upper left")
    leg2 = plt.legend(handles=handles_models, title="Model (edge color)", loc="lower left")
    plt.gca().add_artist(leg1)  # 讓兩個圖例同時顯示

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("Saved:", save_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs", help="實驗根資料夾")
    ap.add_argument("--model", choices=["rnn","lstm","all"], default="all")
    ap.add_argument("--metric", choices=["train","val"], default="train",
                    help="使用訓練或驗證 loss 來上色")
    ap.add_argument("--stat", choices=["final","best"], default="final",
                    help="final=最後一個 epoch；best=歷程中的最小 loss")
    ap.add_argument("--save", default="figs/param_sweep.png")
    ap.add_argument("--min-layers", type=int, default=None, help="只繪製層數 >= 此值的實驗")
    args = ap.parse_args()

    model_filters = [None] if args.model == "all" else [args.model]
    for mf in model_filters:
        pts = collect_points(args.root,
                            model_filter=mf,                # ← 用 mf
                            which=args.metric, stat=args.stat,
                            min_layers=args.min_layers,     # ← 這樣就不會報錯
                            verbose=True)
        tag = (mf.upper() if mf else "RNN+LSTM")
        title = f"{tag} | {args.metric.capitalize()} Loss (BPC) vs Hidden/SeqLen ({args.stat})"
        save = args.save if args.model == "all" else os.path.splitext(args.save)[0] + f"_{mf}.png"
        plot(pts, title, save)
if __name__ == "__main__":
    main()