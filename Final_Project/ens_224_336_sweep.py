import pandas as pd
import numpy as np

p224 = pd.read_csv("L14_224_kfold_prob.csv").sort_values("filename").reset_index(drop=True)
p336 = pd.read_csv("L14_336_kfold_prob.csv").sort_values("filename").reset_index(drop=True)

assert (p224["filename"].values == p336["filename"].values).all()

weights = [(0.6,0.4),(0.5,0.5),(0.4,0.6),(0.3,0.7)]
thresholds = [0.42,0.44,0.46,0.48,0.50]  # 先掃這幾個就好

for w224, w336 in weights:
    p = w224*p224["prob_fake"].values + w336*p336["prob_fake"].values
    for thr in thresholds:
        out = pd.DataFrame({
            "filename": p224["filename"],
            "label": np.where(p >= thr, "fake", "real")
        })
        out.to_csv(f"sub_ens_w{w224:.1f}-{w336:.1f}_thr{thr:.2f}.csv", index=False)

print("done")
