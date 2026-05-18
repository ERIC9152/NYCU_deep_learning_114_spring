import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_prefix", required=True, help="例如 feats_L14_336")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_oof", default="oof_prob.csv")
    ap.add_argument("--out_test", default="test_kfold_prob.csv")
    args = ap.parse_args()

    X = np.load(f"{args.feat_prefix}_X_train.npy")
    y = np.load(f"{args.feat_prefix}_y_train.npy")
    name_tr = np.load(f"{args.feat_prefix}_name_train.npy", allow_pickle=True).astype(str)

    Xte = np.load(f"{args.feat_prefix}_X_test.npy")
    name_te = np.load(f"{args.feat_prefix}_name_test.npy", allow_pickle=True).astype(str)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = np.zeros(len(y), dtype=np.float32)
    test_probs = []

    for f, (tr, va) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:,1].astype(np.float32)
        test_probs.append(clf.predict_proba(Xte)[:,1].astype(np.float32))
        print(f"[F{f}] done")

    test_mean = np.mean(np.stack(test_probs, 0), 0).astype(np.float32)

    pd.DataFrame({"filename": name_tr, "y_true": y, "prob_fake": oof}).to_csv(args.out_oof, index=False)
    pd.DataFrame({"filename": name_te, "prob_fake": test_mean}).to_csv(args.out_test, index=False)

    print("[OK] oof ->", args.out_oof)
    print("[OK] test->", args.out_test)

if __name__ == "__main__":
    main()
