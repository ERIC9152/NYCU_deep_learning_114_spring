#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLIP + MLP/Linear classifier + 強泛化訓練（可交 Kaggle submission）

特點：
- CLIP encoder 凍結，只訓練小 head（線性或 MLP）
- 不需要 fake&real 各半：用 class weight / weighted sampler 抗不平衡
- 強泛化：label smoothing + mixup(embedding) + cosine lr + early stopping
- 推論：簡單 TTA（原圖 + 水平翻）取平均

輸出：submission_final.csv (filename,label)，label ∈ {real,fake}

安裝（建議在 conda env 內）：
pip install torch torchvision pillow numpy pandas scikit-learn open_clip_torch

若沒 tqdm 也能跑（程式已自動降級）
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# tqdm optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(p: Path) -> List[Path]:
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])


def stem_name(p: Path) -> str:
    return p.stem


# -------- CLIP (open_clip) --------
def load_openclip(device: torch.device, model_name: str = "ViT-B-32"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def clip_encode_batch(model, x: torch.Tensor) -> torch.Tensor:
    feat = model.encode_image(x)
    feat = F.normalize(feat, dim=-1)
    return feat


# -------- Dataset --------
class ImagePathsDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int] | None, preprocess, tta: bool = False):
        self.paths = paths
        self.labels = labels
        self.preprocess = preprocess
        self.tta = tta

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")

        x = self.preprocess(img)  # (3,H,W)

        if self.tta:
            # 簡單 TTA：原圖 + 水平翻
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            x2 = self.preprocess(img_flip)
            return x, x2, str(p)

        if self.labels is None:
            return x, str(p)

        y = self.labels[idx]
        return x, y


# -------- Head models --------
class LinearHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# -------- Loss: BCE with label smoothing (and optional focal) --------
def bce_with_logits_label_smoothing(logits, targets, smoothing=0.05):
    # targets: 0/1
    t = targets.float()
    t = t * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, t)


def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0, smoothing=0.05):
    t = targets.float()
    t = t * (1.0 - smoothing) + 0.5 * smoothing
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.where(t >= 0.5, p, 1 - p)
    w = alpha * (1 - pt).pow(gamma)
    return (w * ce).mean()


# -------- Mixup in embedding space --------
def mixup_embeddings(z, y, alpha=0.2):
    if alpha <= 0:
        return z, y.float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(z.size(0), device=z.device)
    z2 = z[idx]
    y2 = y[idx].float()
    z_mix = lam * z + (1 - lam) * z2
    y_mix = lam * y.float() + (1 - lam) * y2
    return z_mix, y_mix


# -------- Train/Eval --------
def train_one_epoch(
    clip_model, head, loader, opt, device,
    mixup_alpha=0.2,
    use_focal=True,
    smoothing=0.05
):
    head.train()
    total_loss = 0.0
    n = 0

    for x, y in tqdm(loader, desc="train", ncols=100):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            z = clip_encode_batch(clip_model, x)

        # mixup on embeddings (fast)
        z_mix, y_mix = mixup_embeddings(z, y, alpha=mixup_alpha)

        logits = head(z_mix)

        if use_focal:
            loss = focal_bce_with_logits(logits, y_mix, alpha=0.25, gamma=2.0, smoothing=smoothing)
        else:
            # y_mix is float in [0,1]
            loss = F.binary_cross_entropy_with_logits(logits, y_mix)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_auc(clip_model, head, loader, device):
    head.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="valid", ncols=100):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = clip_encode_batch(clip_model, x)
        logits = head(z)
        p = torch.sigmoid(logits)
        ys.append(y.cpu().numpy())
        ps.append(p.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return auc


@torch.no_grad()
def predict_tta(clip_model, head, loader, device) -> Tuple[List[str], np.ndarray]:
    head.eval()
    names = []
    probs = []
    for x1, x2, pathstr in tqdm(loader, desc="predict", ncols=100):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        z1 = clip_encode_batch(clip_model, x1)
        z2 = clip_encode_batch(clip_model, x2)

        p1 = torch.sigmoid(head(z1))
        p2 = torch.sigmoid(head(z2))

        p = 0.5 * (p1 + p2)

        probs.append(p.cpu().numpy())
        names.extend([stem_name(Path(s)) for s in pathstr])

    probs = np.concatenate(probs).astype(np.float32)
    return names, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True, help="train 根目錄（含 real/ fake）")
    ap.add_argument("--test_dir", type=str, required=True, help="test 圖片目錄")
    ap.add_argument("--out_csv", type=str, default="submission_final.csv")

    ap.add_argument("--clip_model", type=str, default="ViT-B-32")
    ap.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--use_focal", action="store_true", help="啟用 focal loss（通常更泛化）")

    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--early_stop", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="輸出 fake 的門檻（sigmoid 機率）")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    real_paths = list_images(train_dir / "real")
    fake_paths = list_images(train_dir / "fake")
    test_paths = list_images(test_dir)

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise FileNotFoundError("train_dir 內需要有 real/ 與 fake/ 子資料夾且包含圖片")
    if len(test_paths) == 0:
        raise FileNotFoundError("test_dir 內找不到圖片")

    print(f"[INFO] real={len(real_paths)} fake={len(fake_paths)} test={len(test_paths)} device={device}")

    clip_model, preprocess = load_openclip(device=device, model_name=args.clip_model)

    # labels: fake=1, real=0（習慣用 1 表示正類：fake）
    X_paths = real_paths + fake_paths
    y = np.array([0]*len(real_paths) + [1]*len(fake_paths), dtype=np.int64)

    train_idx, val_idx = train_test_split(
        np.arange(len(X_paths)),
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y
    )

    train_paths = [X_paths[i] for i in train_idx]
    val_paths   = [X_paths[i] for i in val_idx]
    y_train = y[train_idx].tolist()
    y_val   = y[val_idx].tolist()

    # WeightedRandomSampler：不用強迫 1:1，但讓 fake 不會被淹沒
    # 權重 ~ 1 / class_count
    c0 = (np.array(y_train) == 0).sum()
    c1 = (np.array(y_train) == 1).sum()
    w0 = 1.0 / max(c0, 1)
    w1 = 1.0 / max(c1, 1)
    sample_w = [w1 if yy == 1 else w0 for yy in y_train]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    ds_train = ImagePathsDataset(train_paths, y_train, preprocess, tta=False)
    ds_val   = ImagePathsDataset(val_paths, y_val, preprocess, tta=False)
    ds_test  = ImagePathsDataset(test_paths, labels=None, preprocess=preprocess, tta=True)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                          num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

    # head
    # 先用一個 dummy forward 抓 embedding dim
    with torch.no_grad():
        x0, _ = ds_val[0]
        z0 = clip_encode_batch(clip_model, x0.unsqueeze(0).to(device))
        dim = z0.shape[-1]

    if args.head == "linear":
        head = LinearHead(dim)
    else:
        head = MLPHead(dim, hidden=args.hidden, dropout=args.dropout)

    head = head.to(device)

    # optimizer + cosine
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    best_auc = -1.0
    bad = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            clip_model, head, dl_train, opt, device,
            mixup_alpha=args.mixup_alpha,
            use_focal=args.use_focal,
            smoothing=args.label_smoothing
        )
        auc = eval_auc(clip_model, head, dl_val, device)
        sched.step()

        print(f"[E{epoch:02d}] loss={loss:.4f} val_auc={auc:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        # early stop by AUC
        if np.isfinite(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.early_stop:
                print("[INFO] Early stopping triggered.")
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    # predict with TTA, output prob(fake)
    names, prob_fake = predict_tta(clip_model, head, dl_test, device)

    # threshold -> label
    labels = np.where(prob_fake >= float(args.threshold), "fake", "real")

    # 在這裡選擇用機率 (prob_fake) 判斷標籤
    sub = pd.DataFrame({
        "filename": names,
        "label": np.where(prob_fake >= args.threshold, "fake", "real")
    })

    sub.to_csv(args.out_csv, index=False)

    print(f"[OK] Saved {args.out_csv} | fake_ratio={np.mean(labels=='fake'):.3f} | best_val_auc={best_auc:.4f}")


if __name__ == "__main__":
    main()
