import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)

def focal_bce_logits(logits, targets, alpha=0.25, gamma=2.0, smoothing=0.08):
    t = targets.float()
    t = t*(1.0-smoothing) + 0.5*smoothing
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = torch.where(t>=0.5, p, 1-p)
    w = alpha * (1-pt).pow(gamma)
    return (w*ce).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out_prob_csv", type=str, default="test_prob.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    data = np.load(args.npz, allow_pickle=True)
    X = data["X_train"].astype(np.float32)
    y = data["y"].astype(np.int64)
    test_names = data["test_names"]
    Xt = data["X_test"].astype(np.float32)
    Xt2 = data["X_test_flip"].astype(np.float32)

    # Test TTA feature average (embedding-level TTA)
    Xt = 0.5*(Xt + Xt2)

    n, dim = X.shape
    print(f"[INFO] Loaded X={X.shape} Xt={Xt.shape} device={device}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    test_prob_acc = np.zeros((Xt.shape[0],), dtype=np.float64)
    oof_prob = np.zeros((n,), dtype=np.float64)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold}/{args.folds} ===")
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        head = MLPHead(dim, hidden=args.hidden, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs,1))

        # torch tensors
        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).to(device)
        Xt_t  = torch.from_numpy(Xt).to(device)

        best_auc = -1
        best_state = None
        bad = 0

        for ep in range(1, args.epochs+1):
            head.train()
            # shuffle indices (fast)
            idx = torch.randperm(Xtr_t.size(0), device=device)

            # mini-batch
            total_loss = 0.0
            for i in range(0, idx.numel(), args.batch_size):
                bidx = idx[i:i+args.batch_size]
                z = Xtr_t[bidx]
                yt = ytr_t[bidx]

                logits = head(z)
                loss = focal_bce_logits(logits, yt, alpha=0.25, gamma=2.0, smoothing=0.08)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += loss.item()

            # valid AUC
            head.eval()
            with torch.no_grad():
                pv = torch.sigmoid(head(Xva_t)).detach().cpu().numpy()
            auc = roc_auc_score(yva, pv)
            sched.step()
            print(f"E{ep:02d} | Loss: {total_loss:.4f} | Val AUC: {auc:.4f}")

            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 3:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)

        # save oof + test prob
        head.eval()
        with torch.no_grad():
            oof_prob[va] = torch.sigmoid(head(Xva_t)).detach().cpu().numpy()
            test_prob = torch.sigmoid(head(Xt_t)).detach().cpu().numpy()

        test_prob_acc += test_prob / args.folds
        print(f"[Fold {fold}] best_val_auc={best_auc:.4f}")

        # 釋放一點點（避免 fold 後段變慢）
        del head, opt, sched
        torch.cuda.empty_cache()

    # write test prob
    out = pd.DataFrame({"filename": test_names, "prob_fake": test_prob_acc.astype(np.float32)})
    out.to_csv(args.out_prob_csv, index=False)

    # also report overall oof auc (參考用)
    try:
        auc_all = roc_auc_score(y, oof_prob)
        print(f"\n[OK] OOF AUC = {auc_all:.4f}")
    except Exception:
        pass

    print(f"[OK] Saved: {args.out_prob_csv}")

if __name__ == "__main__":
    main()
