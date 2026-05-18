import pandas as pd
import numpy as np

# 讀取剛剛跑出來的兩個檔案
df1 = pd.read_csv("submission_L336.csv")    # 大模型 (凍結)
df2 = pd.read_csv("submission_siglip.csv")  # 中模型 (微調)

# 確保順序一致
df1 = df1.sort_values("filename").reset_index(drop=True)
df2 = df2.sort_values("filename").reset_index(drop=True)

# 簡單平均 (50% + 50%)
final_prob = (df1["prob"] + df2["prob"]) / 2.0

# 輸出最終結果
df1["prob"] = final_prob
df1["label"] = np.where(final_prob >= 0.45, 1, 0)
df1.to_csv("submission_ensemble_thr0.45.csv", index=False)

print("恭喜！最終合體檔案已產生：submission_ensemble_thr0.45.csv")