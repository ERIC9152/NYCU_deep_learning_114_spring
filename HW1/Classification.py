import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# access the data
raw_data = pd.read_csv("2025_ionosphere_data.csv", header=None) 

# 假設資料集的特徵數量是 34
X = raw_data.iloc[:, :-1].values
Y = raw_data.iloc[:, -1].values

# g->1, b->0
Y = (Y == 'g').astype(int)

# store training sample/label and test sample/label properly
np.random.seed(0)
shuffled_indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_data, test_data = shuffled_indices[:train_size], shuffled_indices[train_size:]
X_train, X_test = X[train_data], X[test_data]
Y_train, Y_test = Y[train_data], Y[test_data]

# 確保 Y_train 和 Y_test 是 (N,) 的向量
Y_train = Y_train.reshape(-1)
Y_test  = Y_test.reshape(-1)

# normalize traing data
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# avoid the std to be 0
train_std[train_std == 0] = 1

# 用訓練集的統計量標準化兩個集合
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

#define activation function
def ReLU(x):
    return np.maximum(0,x)

def tanh(x):
    x = np.array(x, dtype=np.float64)
    return np.tanh(x)

def sigmoid(x):
    x = np.array(x, dtype=np.float64)
    # 使用數值穩定的 Sigmoid 函數
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))) 

#define derivative of activation function
def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    t = tanh(x)
    return 1-t**2

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# define the structure and parameter
input_layer = X_train.shape[1] 
hidden_layer = 4 # 降維讓平面上的分類明顯
output_layer = 1 # Binary Classification: 1 output node
lr = 4e-3
epochs = 10000
hidden_act = "tanh"
# define dimansion in each layer 
layer_sizes = [input_layer, 128, 32, 2, hidden_layer, output_layer]  

W, b = [], []
for i in range(len(layer_sizes)-1):
    fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
    is_output = (i == len(layer_sizes)-2)
    
    # 使用 He/Kaiming 初始化 (for ReLU)
    if hidden_act == "ReLU" and not is_output:  
        W.append(np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in))
        b.append(np.full((1, fan_out), 0.01))
    
    # 使用 Xavier/Glorot 初始化 (或 He/Kaiming)
    else: 
        W.append(np.random.randn(fan_in, fan_out) * np.sqrt(1.0/fan_in)) 
        b.append(np.zeros((1, fan_out)))

def activation(x, hidden_act):
    if hidden_act == "tanh":
        return tanh(x)
    elif hidden_act == "sigmoid":
        return sigmoid(x)
    elif hidden_act == "ReLU":
        return ReLU(x)

def forward_pass(x):
    Zs = [] # Zs 儲存線性輸出的值 (logits)
    As = [x] # As 儲存激勵函數的輸出 (activation)
    n_layers = len(W)
    
    # hidden layers
    for i in range(n_layers-1):  
        Z = np.dot(As[-1], W[i]) + b[i]
        Zs.append(Z)
        A = activation(Z, hidden_act) 
        As.append(A)
        
    # output layers：linear output (Logits)
    Z = np.dot(As[-1], W[-1]) + b[-1]
    Zs.append(Z)
    
    # Sigmoid for activation function (for Binary Classification)
    out = sigmoid(Z) 
    
    return Zs, As, out

def activation_derivative(x, hidden_act):
    if hidden_act == "tanh":
        return tanh_derivative(x)
    elif hidden_act == "sigmoid":
        return sigmoid_derivative(x)
    elif hidden_act == "ReLU":
        return ReLU_derivative(x)

def backward_pass(X, y, Zs, As, out):
    n_layers = len(W)
    dW = [None] * n_layers
    db = [None] * n_layers
    m = X.shape[0]
    
    # Output Layer
    dZ = (out.squeeze() - y).reshape(-1, 1) / m  
    
    dW[-1] = np.dot(As[-1].T, dZ)
    db[-1] = np.sum(dZ, axis=0, keepdims=True)
    dA = np.dot(dZ, W[-1].T)
    
    # 隱藏層反向遞歸
    for i in reversed(range(n_layers - 1)):
        dZ = dA * activation_derivative(Zs[i], hidden_act)
        dW[i] = np.dot(As[i].T, dZ)
        db[i] = np.sum(dZ, axis=0, keepdims=True)
        if i > 0:
            dA = np.dot(dZ, W[i].T)
            
    return dW, db

loss_curve = []
train_error_curve, test_error_curve = [], [] 
train_losses, test_losses = [], []

# 潛在特徵記錄
latent_features_history = {}
target_epochs = [10, 500, 1000, 5000, 10000] 


for epoch in range(epochs):
    Zs, As, out_train = forward_pass(X_train)
    
    current_epoch = epoch + 1
    if current_epoch in target_epochs:
        # As[-2] 是倒數第二層 (2D 潛在特徵層) 的輸出 A (ReLU 激活後的結果)
        latent_features_history[current_epoch] = As[-2].copy()

    # Binary Cross-Entropy Loss
    epsilon = 1e-12 #avoid log(0)
    out_sq_train = out_train.squeeze()
    train_loss = -np.mean(Y_train * np.log(out_sq_train + epsilon) + (1 - Y_train) * np.log(1 - out_sq_train + epsilon)) 
    train_losses.append(train_loss)
    
    loss = train_loss 
    loss_curve.append(loss)
    
    # 訓練錯誤率 (Train Error Rate)
    y_pred_train_binary = (out_sq_train > 0.5).astype(int)
    error_rate_train = np.mean(y_pred_train_binary != Y_train)
    train_error_curve.append(error_rate_train)
    
    # 反向傳播
    dW, db = backward_pass(X_train, Y_train, Zs, As, out_train)

    # 梯度下降更新參數
    for i in range(len(W)):
        W[i] =W[i] - lr * dW[i]
        b[i] =b[i] - lr * db[i]

    # 計算測試集損失（用於繪製學習曲線）
    Zs_test, As_test, out_test = forward_pass(X_test)
    out_sq_test = out_test.squeeze()
    test_loss = -np.mean(Y_test * np.log(out_sq_test + epsilon) + (1 - Y_test) * np.log(1 - out_sq_test + epsilon))
    test_losses.append(test_loss) 

    if (epoch + 1) % 500 == 0:  
         print(f"Epoch {epoch+1:4d} | Loss: {loss:.4f} | Train Error Rate: {error_rate_train:.4f}")

    if (epoch + 1) % 500 == 0:
        lr = lr *0.65  # 學習率衰減
        print(f"Epoch {epoch+1} | New LR = {lr:.6f}")


# 最終預測與性能評估
Zs_train, As_train, out_train = forward_pass(X_train)
Zs_test, As_test, out_test = forward_pass(X_test)
out_sq_train = out_train.squeeze()
out_sq_test = out_test.squeeze()

# 訓練集
y_pred_train = (out_sq_train > 0.5).astype(int)
train_error = np.mean(y_pred_train != Y_train)

# 測試集
y_pred_test = (out_sq_test > 0.5).astype(int)
test_error  = np.mean(y_pred_test != Y_test)
loss = -np.mean(Y_train * np.log(out_sq_train + epsilon) + (1 - Y_train) * np.log(1 - out_sq_train + epsilon)) 

print("-" * 50)
print(f"Architecture: {layer_sizes}")
print(f"Train Error Rate = {train_error:.4f} (Accuracy: {1-train_error:.4f})")
print(f"Test  Error Rate = {test_error:.4f} (Accuracy: {1-test_error:.4f})")
print(f"Loss = {loss:.4f}")

# 潛在特徵分佈圖
if latent_features_history:
    
    n_plots = len(target_epochs)
    rows = (n_plots + 1) // 2 
    cols = 2
    plt.figure(figsize=(12, 6 * rows)) 

    for idx, epoch in enumerate(target_epochs):
        if epoch in latent_features_history:
            latent_features = latent_features_history[epoch].copy()
            x_dim = latent_features[:, 0]
            y_dim = latent_features[:, 1]
            class_g_indices = (Y_train == 1) 
            class_b_indices = (Y_train == 0) 
            plt.subplot(rows, cols, idx + 1)
            # Class g (Good, 1) - 藍色
            plt.scatter(x_dim[class_g_indices], y_dim[class_g_indices], label='Class g (Good)', color='blue', alpha=0.5)
            # Class b (Bad, 0) - 紅色
            plt.scatter(x_dim[class_b_indices], y_dim[class_b_indices], label='Class b (Bad)', color='red', alpha=0.5)
            plt.title(f'Latent Features Distribution at {epoch} Epochs')
            plt.legend()

    plt.tight_layout()
    plt.savefig("LatantFeatureDistribution16")

# plot loss curve
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title(f'Learning Curve ({layer_sizes[0]}-{layer_sizes[1]}-{layer_sizes[2]}-{layer_sizes[3]}-{layer_sizes[4]}-{layer_sizes[5]})')
plt.xlabel('Epochs')
plt.ylabel('Loss (Binary Cross-Entropy)')
plt.legend()
plt.grid()
plt.show() 

# plot error rate curve
plt.figure(figsize=(10, 5))
plt.plot(train_error_curve)
plt.xlabel("Epoch")
plt.ylabel("Train Error Rate")
plt.title(f"Training Error Rate Curve ({layer_sizes[0]}-{layer_sizes[1]}-{layer_sizes[2]}-{layer_sizes[3]}-{layer_sizes[4]})")
plt.grid()
plt.show() 
