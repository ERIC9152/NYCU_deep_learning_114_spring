#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images(p: Path) -> List[Path]:
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])

def stem_name(p: Path) -> str:
    return p.stem

# -------- 模型載入與解凍邏輯 --------
def load_openclip_finetune(device: torch.device, model_name: str, unfreeze_layers: int):
    import open_clip
    # 可自定義模型，例如 "ViT-B-32", "ViT-L-14", "vit_so400m_patch14_siglip_384"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai" if "siglip" not in model_name else "timm"
    )
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if unfreeze_layers > 0:
        # 嘗試解凍 Transformer 結構
        visual = model.visual
        if hasattr(visual, 'transformer'): # ViT 架構
            resblocks = visual.transformer.resblocks
            for i in range(len(resblocks) - unfreeze_layers, len(resblocks)):
                for p in resblocks[i].parameters(): p.requires_grad = True
        elif hasattr(visual, 'blocks'): # SigLIP/timm 架構
            for i in range(len(visual.blocks) - unfreeze_layers, len(visual.blocks)):
                for p in visual.blocks[i].parameters(): p.requires_grad = True
            
    return model, preprocess

# -------- Dataset --------
class ImagePathsDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int] | None, preprocess, tta: bool = False):
        self.paths = paths
        self.labels = labels
        self.preprocess = preprocess
        self.tta = tta

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.preprocess(img)
        if self.tta:
            x_flip = self.preprocess(img.transpose(Image.FLIP_LEFT_RIGHT))
            return x, x_flip, str(p)
        return (x, self.labels[idx]) if self.labels is not None else (x, str(p))

class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, z): return self.net(z).squeeze(-1)

# -------- 訓練與損失函數 --------
def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0, smoothing=0.05):
    t = targets.float() * (1.0 - smoothing) + 0.5 * smoothing
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.where(t >= 0.5, p, 1 - p)
    w = alpha * (1 - pt).pow(gamma)
    return (w * ce).mean()

def train_one_epoch(clip_model, head, loader, opt, device, mixup_alpha=0.2):
    clip_model.train(); head.train()
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, desc="  Train"):
        x, y = x.to(device), y.to(device)
        z = F.normalize(clip_model.encode_image(x), dim=-1)
        
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(z.size(0), device=z.device)
            z = lam * z + (1 - lam) * z[idx]
            y = lam * y.float() + (1 - lam) * y[idx].float()

        loss = focal_bce_with_logits(head(z), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0); n += x.size(0)
    return total_loss / n

@torch.no_grad()
def eval_auc(clip_model, head, loader, device):
    clip_model.eval(); head.eval()
    ys, ps = [], []
    for x, y in loader:
        z = F.normalize(clip_model.encode_image(x.to(device)), dim=-1)
        ps.append(torch.sigmoid(head(z)).cpu().numpy())
        ys.append(y.numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--clip_model", type=str, default="ViT-B-32", help="模型名稱")
    ap.add_argument("--n_folds", type=int, default=5, help="K-Fold 數量")
    ap.add_argument("--unfreeze_layers", type=int, default=2)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # 1. 資料加載
    train_dir = Path(args.train_dir)
    real_paths, fake_paths = list_images(train_dir/"real"), list_images(train_dir/"fake")
    X = np.array(real_paths + fake_paths)
    Y = np.array([0]*len(real_paths) + [1]*len(fake_paths))
    
    # 2. K-Fold 訓練迴圈
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    all_test_probs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
        print(f"\n=== Fold {fold+1}/{args.n_folds} ===")
        clip_model, preprocess = load_openclip_finetune(device, args.clip_model, args.unfreeze_layers)
        
        # 獲取 Feature Dim
        with torch.no_grad():
            dummy = preprocess(Image.new('RGB', (224, 224))).unsqueeze(0).to(device)
            dim = clip_model.encode_image(dummy).shape[-1]
        
        head = MLPHead(dim=dim).to(device)
        opt = torch.optim.AdamW([
            {'params': head.parameters(), 'lr': args.lr_head},
            {'params': [p for p in clip_model.parameters() if p.requires_grad], 'lr': args.lr_backbone}
        ], weight_decay=1e-4)
        
        # DataLoader
        y_tr = Y[tr_idx]
        w = [1.0/np.sum(y_tr==0) if l==0 else 1.0/np.sum(y_tr==1) for l in y_tr]
        dl_train = DataLoader(ImagePathsDataset(X[tr_idx].tolist(), y_tr.tolist(), preprocess), 
                              batch_size=args.batch_size, sampler=WeightedRandomSampler(w, len(w)), num_workers=4)
        dl_val = DataLoader(ImagePathsDataset(X[val_idx].tolist(), Y[val_idx].tolist(), preprocess), batch_size=args.batch_size)

        best_auc = 0
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(clip_model, head, dl_train, opt, device)
            auc = eval_auc(clip_model, head, dl_val, device)
            print(f"  E{epoch} | Loss: {loss:.4f} | Val AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save({'clip': clip_model.state_dict(), 'head': head.state_dict()}, ckpt_dir/f"best_fold{fold}.pt")

    # 3. 集成推論 (Ensemble Inference)
    print("\n=== Starting Ensemble Inference ===")
    test_paths = list_images(Path(args.test_dir))
    # 這裡重新載入 preprocess 是為了確保一致性
    _, preprocess = load_openclip_finetune(device, args.clip_model, 0)
    dl_test = DataLoader(ImagePathsDataset(test_paths, None, preprocess, tta=True), batch_size=args.batch_size)
    
    final_probs = np.zeros(len(test_paths))
    for fold in range(args.n_folds):
        ckpt = torch.load(ckpt_dir/f"best_fold{fold}.pt", map_location=device)
        clip_model.load_state_dict(ckpt['clip']); head.load_state_dict(ckpt['head'])
        clip_model.eval(); head.eval()
        
        fold_probs = []
        names = []
        for x, x_flip, pathstr in tqdm(dl_test, desc=f"  Predicting Fold {fold+1}"):
            z = F.normalize(clip_model.encode_image(x.to(device)), dim=-1)
            z_f = F.normalize(clip_model.encode_image(x_flip.to(device)), dim=-1)
            p = 0.5 * (torch.sigmoid(head(z)) + torch.sigmoid(head(z_f)))
            fold_probs.append(p.cpu().numpy())
            if fold == 0: names.extend([stem_name(Path(s)) for s in pathstr])
            
        final_probs += np.concatenate(fold_probs) / args.n_folds

    # 4. 輸出結果
    pd.DataFrame({"filename": names, "label": np.where(final_probs >= 0.5, "fake", "real")}).to_csv("submission_ensemble.csv", index=False)
    print(f"\n[OK] Saved submission_ensemble.csv with {args.n_folds} folds ensemble.")

if __name__ == "__main__":
    main()