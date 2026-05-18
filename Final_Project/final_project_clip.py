#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP + One-Class (只用 real 訓練) 產生 submission。

流程：
1) 用 CLIP 抽 embedding
2) 只用 train/real 擬合分佈（LedoitWolf 協方差）
3) 用 Mahalanobis distance 當異常分數
4) 閾值：用 real holdout 的分位數 q（例如 0.995）
   score > thr => fake else real

輸出：submission_final.csv (filename,label), label ∈ {real,fake}

依賴：
pip install torch torchvision pillow numpy pandas scikit-learn tqdm open_clip_torch
（若 open_clip 裝不起來，可改用 transformers 版本，見下方註解）
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(p: Path) -> List[Path]:
    if not p.exists():
        return []
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])


def stem_name(p: Path) -> str:
    return p.stem


def load_clip_openclip(device: torch.device, model_name: str = "ViT-B-32"):
    import open_clip  # open_clip_torch
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images_openclip(
    model,
    preprocess,
    paths: List[Path],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP encode", ncols=100):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        x = torch.stack([preprocess(img) for img in imgs]).to(device)
        f = model.encode_image(x)
        f = F.normalize(f, dim=-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def fit_real_distribution(X_real: np.ndarray):
    lw = LedoitWolf().fit(X_real)
    mean = lw.location_.astype(np.float32)
    precision = lw.precision_.astype(np.float32)  # inv(cov)
    return mean, precision


def mahalanobis_scores(X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    d = X - mean[None, :]
    m2 = np.einsum("nd,dd,nd->n", d, precision, d)
    m2 = np.maximum(m2, 0.0)
    return np.sqrt(m2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True, help="訓練資料根目錄，內含 real/ fake")
    ap.add_argument("--test_dir", type=str, required=True, help="測試資料目錄（全是圖片）")
    ap.add_argument("--out_csv", type=str, default="submission_final.csv")
    ap.add_argument("--model", type=str, default="ViT-B-32", help="open_clip model name")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--quantile", type=float, default=0.995, help="用 real holdout 分位數設閾值")
    ap.add_argument("--real_holdout", type=float, default=0.15, help="real 留作閾值估計比例")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    real_dir = train_dir / "real"
    real_paths = list_images(real_dir)
    if len(real_paths) == 0:
        raise FileNotFoundError(f"找不到 real 圖片：{real_dir}")

    test_paths = list_images(test_dir)
    if len(test_paths) == 0:
        raise FileNotFoundError(f"找不到 test 圖片：{test_dir}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, preprocess = load_clip_openclip(device=device, model_name=args.model)

    print(f"[INFO] real={len(real_paths)} test={len(test_paths)} device={device}")

    # --- split real into fit / holdout (只用 real) ---
    rng = np.random.default_rng(42)
    idx = np.arange(len(real_paths))
    rng.shuffle(idx)
    n_hold = int(len(idx) * float(args.real_holdout))
    n_hold = max(200, min(n_hold, len(idx)//2))
    hold_idx = idx[:n_hold]
    fit_idx = idx[n_hold:]

    real_fit = [real_paths[i] for i in fit_idx]
    real_hold = [real_paths[i] for i in hold_idx]

    # --- embeddings ---
    X_fit = encode_images_openclip(model, preprocess, real_fit, device, args.batch_size)
    X_hold = encode_images_openclip(model, preprocess, real_hold, device, args.batch_size)
    X_test = encode_images_openclip(model, preprocess, test_paths, device, args.batch_size)

    # --- fit one-class distribution ---
    mean, precision = fit_real_distribution(X_fit)
    s_hold = mahalanobis_scores(X_hold, mean, precision)
    thr = float(np.quantile(s_hold, min(max(args.quantile, 0.5), 0.9999)))
    print(f"[INFO] threshold (quantile={args.quantile}) = {thr:.4f}")

    # --- predict ---
    s_test = mahalanobis_scores(X_test, mean, precision)
    labels = np.where(s_test > thr, "fake", "real")

    sub = pd.DataFrame({
        "filename": [stem_name(p) for p in test_paths],
        "label": labels
    })
    sub.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved {args.out_csv} | fake_ratio={np.mean(labels=='fake'):.3f}")


if __name__ == "__main__":
    main()
