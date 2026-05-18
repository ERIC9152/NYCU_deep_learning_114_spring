import pandas as pd

df = pd.read_csv("L14_336_kfold_prob.csv")

for thr in [0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.52,0.55]:
    out = df[["filename"]].copy()
    out["label"] = (df["prob_fake"] >= thr).map({True:"fake", False:"real"})
    out.to_csv(f"sub_thr_{thr:.2f}.csv", index=False)

print("done")
