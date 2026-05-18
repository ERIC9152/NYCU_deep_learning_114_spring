#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Try importing Albumentations
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] Albumentations not found. Using basic augmentations.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def list_images(p: Path) -> List[Path]:
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])

# ==========================================
# Model EMA
# ==========================================
class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            for k in msd:
                esd[k] = self.decay * esd[k] + (1 - self.decay) * msd[k]

# ==========================================
# Model Architecture
# ==========================================
class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)

class CLIPClassifier(nn.Module):
    def __init__(self, model_name: str, device: torch.device, unfreeze_layers: int = 0):
        super().__init__()
        import open_clip
        print(f"[INFO] Loading OpenCLIP: {model_name}")
        backbone, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai" if "siglip" not in model_name else "timm"
        )
        self.backbone = backbone.to(device)
        self.preprocess = preprocess
        
        with torch.no_grad():
            # 1. 自動偵測模型需要的圖片尺寸 (解決 336 報錯問題)
            if hasattr(self.backbone.visual, 'image_size'):
                img_size = self.backbone.visual.image_size
                # 有些版本回傳 tuple (336, 336)，有些回傳 int 336
                if isinstance(img_size, tuple) or isinstance(img_size, list):
                    img_size = img_size[0]
            else:
                # 如果偵測失敗，檢查名字裡有沒有 336
                img_size = 336 if '336' in model_name else 224
            
            print(f"[DEBUG] Model expects image size: {img_size}")
            dummy = torch.zeros(1, 3, img_size, img_size).to(device)

            # 2. 計算特徵維度
            if hasattr(self.backbone, 'encode_image'):
                feat_dim = self.backbone.encode_image(dummy).shape[-1]
            else:
                feat_dim = self.backbone(dummy).shape[-1]
        
        self.head = MLPHead(feat_dim)

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if unfreeze_layers > 0:
            self._unfreeze_visual_layers(unfreeze_layers)
            
    def _unfreeze_visual_layers(self, n):
        visual = self.backbone.visual
        if hasattr(visual, 'transformer'):
            blocks = visual.transformer.resblocks
        elif hasattr(visual, 'blocks'):
            blocks = visual.blocks
        else:
            return

        total = len(blocks)
        print(f"[INFO] Unfreezing last {n} visual blocks (Total: {total})")
        for i in range(total - n, total):
            for p in blocks[i].parameters():
                p.requires_grad = True
        
        if hasattr(visual, 'ln_post'):
            for p in visual.ln_post.parameters(): p.requires_grad = True

    def forward(self, x):
        features = self.backbone.encode_image(x)
        features = F.normalize(features, dim=-1)
        return self.head(features)

# ==========================================
# Dataset
# ==========================================
class AI_Dataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int] = None, preprocess=None, is_train: bool = False, tta: bool = False):
        self.paths = paths
        self.labels = labels
        self.preprocess = preprocess
        self.is_train = is_train
        self.tta = tta

        if HAS_ALBUMENTATIONS and is_train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                    A.MotionBlur(blur_limit=(3, 5), p=0.3),
                    A.ISONoise(p=0.3),
                ], p=0.5),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.aug:
                img_np = np.array(image)
                augmented = self.aug(image=img_np)['image']
                image = Image.fromarray(augmented)
            
            x = self.preprocess(image)
            if self.tta:
                x_flip = self.preprocess(image.transpose(Image.FLIP_LEFT_RIGHT))
                return x, x_flip, str(path)
            return (x, self.labels[idx]) if self.labels is not None else (x, str(path))
        except Exception:
            return torch.zeros(3, 224, 224), 0 if self.labels is not None else str(path)

# ==========================================
# Training & Loss
# ==========================================
def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0, smoothing=0.05):
    t = targets.float() * (1.0 - smoothing) + 0.5 * smoothing
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.where(t >= 0.5, p, 1 - p)
    w = alpha * (1 - pt).pow(gamma)
    return (w * ce).mean()

def train_one_epoch(model, loader, opt, device, scaler, ema=None, mixup_alpha=0.2, accum_steps=1):
    model.train()
    total_loss, n = 0.0, 0
    opt.zero_grad(set_to_none=True)
    
    for i, (x, y) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        x, y = x.to(device), y.to(device)
        
        # Mixup
        if mixup_alpha > 0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(x.size(0)).to(device)
            x_mixed = lam * x + (1 - lam) * x[idx]
            y_mixed = lam * y.float() + (1 - lam) * y[idx].float()
            with torch.amp.autocast('cuda'):
                logits = model(x_mixed)
                loss = focal_bce_with_logits(logits, y_mixed)
        else:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = focal_bce_with_logits(logits, y)

        # Gradient Accumulation Logic
        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema: ema.update(model)

        total_loss += loss.item() * accum_steps * x.size(0)
        n += x.size(0)
        
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))

# ==========================================
# Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--clip_model", type=str, default="ViT-L-14")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--unfreeze", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_bb", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_csv", type=str, default="submission_robust.csv", help="自訂輸出檔名")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    train_root = Path(args.train_dir)
    real_paths = list_images(train_root / "real")
    fake_paths = list_images(train_root / "fake")
    X = np.array(real_paths + fake_paths)
    Y = np.array([0] * len(real_paths) + [1] * len(fake_paths))
    
    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    output_dir = Path("checkpoints_robust")
    output_dir.mkdir(exist_ok=True)
    
    import open_clip
    # Pre-load preprocess to avoid reloading model for dataset creation
    try:
        _, _, base_preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained="openai")
    except:
        # Fallback for some models like siglip/convnext where "openai" pretrained might throw error or need "timm"
        _, _, base_preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained="timm")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
        print(f"\n=== Fold {fold + 1}/{args.folds} ===")
        
        model = CLIPClassifier(args.clip_model, device, unfreeze_layers=args.unfreeze).to(device)
        ema = ModelEMA(model, device=device)
        
        opt = torch.optim.AdamW([
            {'params': model.head.parameters(), 'lr': args.lr},
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': args.lr_bb}
        ], weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

        y_tr = Y[tr_idx]
        weights = 1. / np.bincount(y_tr)
        sampler = WeightedRandomSampler(weights[y_tr], len(y_tr))

        ds_train = AI_Dataset(X[tr_idx], Y[tr_idx], base_preprocess, is_train=True)
        ds_val = AI_Dataset(X[val_idx], Y[val_idx], base_preprocess, is_train=False)

        # num_workers set to 2 to save memory, increase if you have RAM
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        best_auc = 0
        for ep in range(1, args.epochs + 1):
            loss = train_one_epoch(model, dl_train, opt, device, scaler, ema=ema, accum_steps=args.accum_steps)
            scheduler.step()
            val_auc = evaluate(ema.module, dl_val, device)
            print(f"  Ep {ep} | Loss: {loss:.4f} | EMA Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(ema.module.state_dict(), output_dir / f"best_fold_{fold}.pt")
        
        del model, ema, opt, scaler, dl_train, dl_val
        torch.cuda.empty_cache()

    # 3. Ensemble Inference
    print("\n=== Ensemble Inference ===")
    test_paths = list_images(Path(args.test_dir))
    ds_test = AI_Dataset(test_paths, None, base_preprocess, is_train=False, tta=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, num_workers=2)

    model = CLIPClassifier(args.clip_model, device, unfreeze_layers=0).to(device)
    final_probs = np.zeros(len(test_paths))
    filenames = []

    for fold in range(args.folds):
        ckpt_path = output_dir / f"best_fold_{fold}.pt"
        if not ckpt_path.exists(): continue
        
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for x, x_flip, p_strs in tqdm(dl_test, desc=f"Fold {fold+1}"):
                if fold == 0: filenames.extend([Path(p).name for p in p_strs])
                z1 = model(x.to(device))
                z2 = model(x_flip.to(device))
                p = (torch.sigmoid(z1) + torch.sigmoid(z2)) / 2.0
                fold_preds.append(p.cpu().numpy())
        final_probs += np.concatenate(fold_preds)

    final_probs /= args.folds
    pd.DataFrame({"filename": filenames, "label": np.where(final_probs >= 0.5, 1, 0), "prob": final_probs}).to_csv(args.output_csv, index=False)
    print(f"[OK] Saved {args.output_csv}")

if __name__ == "__main__":
    main()