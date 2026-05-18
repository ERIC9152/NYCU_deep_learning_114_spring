import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
#import the data
raw_data = pd.read_csv(r"D:\DL\HW1_StudentID_Name\HW1_StudentID_Name\2025_energy_efficiency_data.csv")

#shuffle data samples for training and for testing
shuffled_indices = np.random.permutation(raw_data.index)
shuffled_data = raw_data.iloc[shuffled_indices].reset_index(drop=True)

#one hot encoding
data = pd.get_dummies(shuffled_data, columns=['Orientation', 'Glazing Area Distribution'])

#store training sample/label and test sample/label properly
train_size = int(0.75 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
feature_cols = [col for col in data.columns if col not in ['Heating Load', 'Cooling Load']]
X_train = train_data[feature_cols].to_numpy()
y_train = train_data['Heating Load'].to_numpy()
X_test = test_data[feature_cols].to_numpy()
y_test = test_data['Heating Load'].to_numpy()

#normalization
X_train = X_train.astype(np.float64)
X_test  = X_test.astype(np.float64)
y_train = y_train.astype(np.float64)
y_test  = y_test.astype(np.float64)

X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0) + 1e-8
X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

#define the structure and parameter
input_layer = X_train.shape[1]
output_layer = 1
lr = 3e-3
epochs = 5000

hidden_act = "ReLU"

print(input_layer)
#define number of hidden layers and neurons 
layer_sizes = [input_layer, 64, 32, output_layer]  

W, b = [], []
for i in range(len(layer_sizes)-1):
    fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
    is_output = (i == len(layer_sizes)-2)
    if not is_output:  # 隱藏層
        if hidden_act == "ReLU":
            W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in))
            b.append(np.full((1, fan_out), 0.01))
        else:  # tanh/sigmoid
            limit = np.sqrt(6.0/(fan_in + fan_out))
            W.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
            b.append(np.zeros((1, fan_out)))
    else:  # 輸出層線性
        # 用 fan_in 的尺度就好
        W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in))
        b.append(np.zeros((1, fan_out)))

def activation(x, hidden_act):
    if hidden_act == "tanh":
        x = np.array(x, dtype=np.float64)
        return np.tanh(x)
    elif hidden_act == "sigmoid":
        x = np.array(x, dtype=np.float64)
        return 1/(1+np.exp(-x))
    elif hidden_act == "ReLU":
        return np.maximum(0, x)

def forward_pass(x):
    Zs = []
    As = [x]
    n_layers = len(W)
    for i in range(n_layers-1):  # 前面各層
        Z = np.dot(As[-1], W[i]) + b[i]
        Zs.append(Z)
        A = activation(Z, hidden_act)  # 或依題目換成你想要的 activation
        As.append(A)
    # 最後一層是「輸出層」無 activation
    Z = np.dot(As[-1], W[-1]) + b[-1]
    Zs.append(Z)
    out = Z
    return Zs, As, out

def activation_derivative(x, hidden_act):
    if hidden_act == "tanh":
        t = np.tanh(x)
        return 1 - t**2
    elif hidden_act == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif hidden_act == "ReLU":
        return np.where(x > 0, 1, 0)

def backward_pass(X, y, Zs, As, out):
    n_layers = len(W)
    dW = [None] * n_layers
    db = [None] * n_layers
    m = X.shape[0]
    # output layer
    dout = 2 * (out.squeeze() - y) / m
    dZ = dout.reshape(-1, 1)  # (m, 1)
    dW[-1] = np.dot(As[-1].T, dZ)
    db[-1] = np.sum(dZ, axis=0, keepdims=True)
    dA = np.dot(dZ, W[-1].T)
    # 隱藏層反向遞歸（倒著來）
    for i in reversed(range(n_layers - 1)):
        dZ = dA * activation_derivative(Zs[i], hidden_act)
        dW[i] = np.dot(As[i].T, dZ)
        db[i] = np.sum(dZ, axis=0, keepdims=True)
        if i > 0:
            dA = np.dot(dZ, W[i].T)
    return dW, db

loss_curve = []
train_rmse_curve, test_rmse_curve = [], []

for epoch in range(epochs):
    Zs, As, out = forward_pass(X_train)
    loss = np.mean((out.squeeze() - y_train)**2)#sum of square error
    loss_curve.append(loss)#sum of square error curve
    rmse_train = np.sqrt(np.mean((out.squeeze() - y_train)**2))
    train_rmse_curve.append(rmse_train)
    
    dW, db = backward_pass(X_train, y_train, Zs, As, out)

    for i in range(len(W)):
        W[i] -= lr * dW[i]
        b[i] -= lr * db[i]

    if epoch % 100 == 0:  
         print(f"Epoch {epoch:4d} | Train RMSE: {rmse_train:.4f} ")

#prediction on the testing set
Zs_train, As_train, out_train = forward_pass(X_train)   # (N_train, 1)
Zs_test, As_test, out_test  = forward_pass(X_test)    # (N_test, 1)
train_RMSE = np.sqrt(np.mean((out_train.squeeze() - y_train)**2))
test_RMSE  = np.sqrt(np.mean((out_test.squeeze() - y_test)**2))
y_pred_train = out_train.squeeze()
y_pred_test  = out_test.squeeze()
x_tr = np.arange(len(y_train))
x_te = np.arange(len(y_test))

print(np.min(y_pred_train), np.max(y_pred_train))
print(np.min(y_pred_test), np.max(y_pred_test))
print(f"Final Train RMSE = {train_RMSE:.4f}")
print(f"Final Test  RMSE = {test_RMSE:.4f}")
print( f"Epoch {epoch:4d} | Loss {loss:.4f} | ")

#plot loss curve
plt.plot(train_rmse_curve)
plt.xlabel("Epoch")
plt.ylabel("Train root mean square error (RMSE)")
plt.title("Training Curve")
plt.grid()
plt.show()

# 畫預測值 vs 真實值
plt.scatter(y_test, out_test.squeeze())
plt.xlabel("True Heating Load")
plt.ylabel("Predicted Heating Load")
plt.title("Test Prediction")
plt.grid()
plt.show()

#Training plot
plt.figure(1)
plt.plot(x_tr, y_train, label="True (Training set)", color="red", linewidth=1)
plt.plot(x_tr, y_pred_train, label="Pred (Training set)", color="blue", linewidth=1)
plt.xlabel("Training Sample Index ")
plt.ylabel("Target")
plt.title("Prediction on Training Data")
plt.legend()
plt.grid()
plt.show()

#Testing plot
plt.figure(2)
plt.plot(x_te, y_test, label="True (Testing set)", color="red", linewidth=1)
plt.plot(x_te, y_pred_test, label="Pred (Testing set)", color="blue", linewidth=1)
plt.xlabel("Testing Sample Index ")
plt.ylabel("Target")
plt.title("Prediction on Test Data")
plt.legend()
plt.grid()
plt.show()

# =========================================================
# (c) Feature Selection — 直接共用你現有的初始化/前傳/反傳
# 需求：X_train, y_train, X_test, y_test, initialize_parameters, forward_pass, backward_pass
# 貼法：把本段貼在檔案最後（你原本的學習曲線與圖都畫完之後）
# =========================================================

feature_names = feature_cols.copy()  

def initialize_parameters(layer_sizes, hidden_activation="ReLU"): 
    W, b = [], []
    for i in range(len(layer_sizes)-1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
        is_output = (i == len(layer_sizes)-2)
        if not is_output:
            if hidden_activation == "ReLU":
                W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in))
                b.append(np.full((1, fan_out), 0.01))
            else:
                limit = np.sqrt(6.0/(fan_in + fan_out))
                W.append(np.random.uniform(-limit, limit, (fan_in, fan_out)))
                b.append(np.zeros((1, fan_out)))
        else:
            W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in))
            b.append(np.zeros((1, fan_out)))
    return W, b

def train_eval_once(Xtr, ytr, Xte, yte,
                    layer_sizes,
                    hidden_act="ReLU",
                    epochs=2000,
                    lr=3e-3,
                    weight_decay=1e-4,
                    verbose=False):
    """
    用你已經定義的 initialize_parameters, forward_pass, backward_pass
    做 full-batch 訓練，回傳 (test_RMSE, (W, b))
    """
    global W, b, hidden_activation
    hidden_activation = hidden_act
    W, b = initialize_parameters(layer_sizes, hidden_activation)

    m = Xtr.shape[0]
    for ep in range(epochs):
        Zs, As, out = forward_pass(Xtr)

        # 回歸：MSE
        loss = np.mean((out - ytr) ** 2)

        # 反傳
        dW, db = backward_pass(Xtr, ytr, Zs, As, out)

        # L2 權重衰退
        if weight_decay > 0:
            for i in range(len(dW)):
                dW[i] += weight_decay * W[i]

        # 參數更新（SGD）
        for i in range(len(W)):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]

        # （可選）更平滑的學習率排程
        # if (ep + 1) % 1000 == 0:
        #     lr *= 0.8

        if verbose and (ep + 1) % 500 == 0:
            _, _, out_te = forward_pass(Xte)
            rmse_tr = float(np.sqrt(loss))
            rmse_te = float(np.sqrt(np.mean((out_te - yte) ** 2)))
            print(f"[ep {ep+1}] RMSE(train)={rmse_tr:.4f}, RMSE(test)={rmse_te:.4f}")

    # 最後評估 Test RMSE
    _, _, out_test = forward_pass(Xte)
    test_RMSE = float(np.sqrt(np.mean((out_test - yte) ** 2)))
    return test_RMSE, (W, b)

def predict_with_params(X, params):
    """用指定 (W,b) 做一次前傳（不重訓練），給 permutation importance 用"""
    global W, b
    W, b = params
    _, _, out = forward_pass(X)
    return out

# ========= 群組建構：把 one-hot 欄位合成一個「特徵群組」 =========
# 用你前面已有的 feature_cols 來建群組
def build_feature_groups(feature_cols):
    groups = []
    used = set()

    # 1) 兩個類別特徵：把對應 one-hot 欄位收成一組
    prefixes = ["Orientation_", "Glazing Area Distribution_"]
    for p in prefixes:
        idxs = [i for i, name in enumerate(feature_cols) if name.startswith(p)]
        if idxs:
            # 群組名字（去掉底線，比較好看）
            gname = p[:-1] if p.endswith("_") else p
            groups.append((gname, idxs))
            used.update(idxs)

    # 2) 其餘連續特徵：各自成為一組（單一欄位群組）
    for i, name in enumerate(feature_cols):
        if i not in used:
            groups.append((name, [i]))

    return groups

groups = build_feature_groups(feature_cols)
input_dim = X_train.shape[1]

# ===== 0) 先做 baseline（全部特徵） =====
layer_sizes_base = [input_dim, 64, 32, 1]
baseline_rmse, baseline_params = train_eval_once(
    X_train, y_train, X_test, y_test,
    layer_sizes=layer_sizes_base,
    hidden_act="ReLU",
    epochs=2000,
    lr=3e-3,
    weight_decay=1e-4,
    verbose=False
)
print(f"\n[FS-Grouped] Baseline Test RMSE (all features) = {baseline_rmse:.4f}")

lofo_grp = []
for gname, gidx in groups:
    keep = [i for i in range(input_dim) if i not in gidx]
    Xtr = X_train[:, keep]
    Xte = X_test[:, keep]
    rmse_out, _ = train_eval_once(
        Xtr, y_train, Xte, y_test,
        layer_sizes=[Xtr.shape[1], 64, 32, 1],
        hidden_act="ReLU",
        epochs=2000,# 稍微減少epoch，雖然RMSE減少，但相對關係不變
        lr=3e-3,
        weight_decay=1e-4,
        verbose=False
    )
    lofo_grp.append((gname, rmse_out, rmse_out - baseline_rmse))

lofo_grp_sorted = sorted(lofo_grp, key=lambda x: x[2], reverse=True)

plt.figure(figsize=(10, 4))
plt.bar(range(len(lofo_grp_sorted)), [d for (_, _, d) in lofo_grp_sorted])
plt.xticks(range(len(lofo_grp_sorted)), [n for (n, _, _) in lofo_grp_sorted], rotation=45, ha='right')
plt.ylabel("Δ Test RMSE (remove a GROUP)")
plt.title("LOFO (Grouped by original features)")
plt.tight_layout()
plt.show()


# Permutation Importance — Grouped (for Regression, RMSE)
def permutation_importance_grouped_regression(
    X, y, params, groups, n_repeats=10, seed=0
):
    """
    以 RMSE 為指標的群組 permutation importance（不重訓，只 forward）
    參數：
      X, y       : 用來評估的重要度資料（建議為測試集）
      params     : (W, b)，已訓練好的 baseline 參數
      groups     : [(group_name, [col_idx, ...]), ...]
      n_repeats  : 每個群組打亂的重複次數（取平均與標準差）
      seed       : 亂數種子
    回傳：
      results_sorted, base_rmse
      results_sorted = [(name, mean_delta_rmse, std_delta_rmse)]，依重要度由大到小
    說明：
      ΔRMSE = RMSE(permuted) - RMSE(base)，越大 ⇒ 越重要
    """
    rng = np.random.default_rng(seed)

    base_pred  = predict_with_params(X.copy(), params)
    base_rmse  = float(np.sqrt(np.mean((base_pred.squeeze() - y) ** 2)))

    results = []
    for gname, gidx in groups:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(X.shape[0])
            # 群組一起用「同一個 perm」打亂，以保留組內結構
            Xp[:, gidx] = X[perm][:, gidx]

            pred   = predict_with_params(Xp, params).squeeze()
            rmse_p = float(np.sqrt(np.mean((pred - y) ** 2)))
            deltas.append(rmse_p - base_rmse)
        deltas = np.asarray(deltas, dtype=float)
        results.append((gname, float(deltas.mean()), float(deltas.std())))

    # 依平均 ΔRMSE 由大到小排序
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted, base_rmse

perm_results, base_rmse = permutation_importance_grouped_regression(
    X_test, y_test, baseline_params, groups, n_repeats=10, seed=0
)

print("\n[Permutation Importance — Grouped | Metric: ΔRMSE ")
print(f"Baseline Test RMSE = {base_rmse:.4f}")
for name, mu, sd in perm_results:
    print(f"{name:>32s}  ΔRMSE={mu:+.4f} ± {sd:.4f}")

# 視覺化：長條圖 + 誤差棒
names = [n for n, _, _ in perm_results]
means = np.array([m for _, m, _ in perm_results])
stds  = np.array([s for _, _, s in perm_results])
order = np.argsort(-means)

plt.figure(figsize=(10, 4))
plt.bar(range(len(names)), means[order], yerr=stds[order], capsize=4)
plt.xticks(range(len(names)), np.array(names)[order], rotation=45, ha='right')
plt.ylabel("Δ Test RMSE ")
plt.title("Permutation Importance — Grouped ")
plt.tight_layout()
plt.show()

