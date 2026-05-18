import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, initializers
from common.utils import set_seed, ensure_dir, param_count, save_history, time_stamp
from common.plotting import plot_learning_curves, plot_weight_bias_hists, plot_examples, plot_feature_maps


set_seed(42)

OUT_DIR = "outputs/cifar10"
EPOCHS = 50
BATCH = 128

EXPERIMENTS = [
    {"name": "k5_wd1e-4_dr0_lr1e-3_s1", "kernel":5, "l2":5e-4, "dropout":0, "lr":0.01, "stride":1},
    {"name": "k5_wd0_dr0_lr1e-3_s1",   "kernel":5, "l2":0,    "dropout":0, "lr":0.01, "stride":1},
]

# ==== Preprocess configuration====
PREPROC = {
    "scale01": False,
    "per_channel_standardize": True,   # ← 打開通道標準化（關鍵）
    "eps": 1e-7,
    "viz_samples": 16,
    "augment": {
        "pad": 4,
        "crop": 32,
        "flip": True,
        "rot_deg": 15,
        "trans": 0.10,
        "zoom": 0.10,
        "brightness": 0.10,            # ← 降低光度擾動
        "contrast": 0.10               # ← 降低對比擾動
    }
}

class TestEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test, self.y_test = x_test, y_test
        self.test_loss, self.test_acc = [], []
    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.test_loss.append(float(loss))
        self.test_acc.append(float(acc))

def preprocess_data(x_train, x_test):
    """
    Keep raw arrays 0..255; compute channel mean/std on [0,1] (train only).
    Scaling / standardization will be done INSIDE the model so train/val/test
    share an identical pipeline.
    """
    x_train = x_train.astype("float32")
    x_test  = x_test.astype("float32")

    # copies for visualization if you need them
    x0_train = x_train.copy()
    x0_test  = x_test.copy()

    # stats on [0,1]
    x01 = x_train / 255.0
    mean = x01.mean(axis=(0,1,2), keepdims=False)
    std  = x01.std(axis=(0,1,2), keepdims=False) + PREPROC["eps"]

    stats = {
        "scale01": PREPROC["scale01"],
        "per_channel_standardize": PREPROC["per_channel_standardize"],
        "mean": mean.tolist(),
        "std":  std.tolist()
    }
    return (x_train, x_test, stats, x0_train, x0_test)

def augment():
    a = []
    # geometric
    a.append(layers.ZeroPadding2D(4))
    a.append(layers.RandomCrop(32, 32))
    a.append(layers.RandomFlip("horizontal"))
    a.append(layers.RandomRotation(15/360.0, fill_mode="reflect"))
    a.append(layers.RandomTranslation(0.10, 0.10, fill_mode="reflect"))
    a.append(layers.RandomZoom((-0.10, 0.10), (-0.10, 0.10), fill_mode="reflect"))
    # photometric (these expect inputs in [0,1])
    a.append(layers.RandomBrightness(0.20))
    a.append(layers.RandomContrast(0.20))
    return tf.keras.Sequential(a)

def _make_grid(images, ncols=8):
    """Make a grid image from NHWC array in [0,1] (auto clip/normalize)."""
    imgs = images.copy()
    if imgs.ndim == 3:  # HWC
        imgs = imgs[None, ...]
    # Normalize each image to [0,1] for safe plotting
    vmin = imgs.min(axis=(1,2,3), keepdims=True)
    vmax = imgs.max(axis=(1,2,3), keepdims=True)
    imgs = (imgs - vmin) / np.clip(vmax - vmin, 1e-8, None)
    n = imgs.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    H, W = imgs.shape[1], imgs.shape[2]
    grid = np.ones((nrows*H, ncols*W, 3), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, ncols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W, :] = imgs[i]
    return grid

def save_preproc_report(out_dir, x0_train, x_train, stats, aug_layer):
    """Save figures and a text report for Section 2-5."""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # 原始 vs. 標準化（從同一批樣本取樣）
    rng = np.random.default_rng(PREPROC.get("viz_seed", 0))
    n = min(PREPROC.get("viz_samples", 16), len(x0_train))
    idx = rng.choice(len(x0_train), n, replace=False)
    raw   = x0_train[idx].astype("float32") / 255.0  # 轉到 [0,1] 以便可視化
    norm  = x_train[idx].copy()

    grid_raw  = _make_grid(raw,  ncols=8)
    grid_norm = _make_grid(norm, ncols=8)

    plt.imsave(os.path.join(out_dir, "a_raw_grid.png"),  np.clip(grid_raw,  0,1))
    plt.imsave(os.path.join(out_dir, "b_standardized_grid.png"), np.clip(grid_norm,0,1))

    # 增強示例（只對同一批 raw 套用 augmentation）
    if len(aug_layer.layers) > 0:
        aug_imgs = aug_layer(raw, training=True).numpy()
        grid_aug = _make_grid(aug_imgs, ncols=8)
        plt.imsave(os.path.join(out_dir, "c_augmented_grid.png"), np.clip(grid_aug,0,1))

    # 可視化 per-channel mean/std（還原到 [0,1] 顯示）
    mean = np.array(stats["mean"]).reshape(1,1,3).astype("float32")
    std  = np.array(stats["std"]).reshape(1,1,3).astype("float32")
    mean_img = np.tile(mean, (32,32,1))
    # 將 mean/std 映射到 0-1 以便觀察（std 只為亮度示意）
    mean_img = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
    std_img  = np.tile(std, (32,32,1))
    std_img  = (std_img - std_img.min()) / (std_img.max() - std_img.min() + 1e-8)
    plt.imsave(os.path.join(out_dir, "d_channel_mean.png"), np.clip(mean_img,0,1))
    plt.imsave(os.path.join(out_dir, "e_channel_std.png"),  np.clip(std_img ,0,1))

    # D) 輸出文字說明
    with open(os.path.join(out_dir, "preprocess.txt"), "w", encoding="utf-8") as f:
        f.write(
            "CIFAR-10 preprocessing (Section 2-5)\n"
            "1) Scaling: {}  → input/255.\n"
            "2) Standardization: {}  → per-channel mean/std computed on training set.\n"
            "   mean(R,G,B) = {}\n"
            "   std (R,G,B) = {}\n"
            "3) Data augmentation (train only): pad={} , crop={} , flip={}.\n"
            "   Files: a_raw_grid.png, b_standardized_grid.png, c_augmented_grid.png,\n"
            "          d_channel_mean.png, e_channel_std.png.\n".format(
                PREPROC["scale01"],
                PREPROC["per_channel_standardize"],
                np.round(np.array(stats['mean']), 6).tolist(),
                np.round(np.array(stats['std']),  6).tolist(),
                PREPROC['augment']['pad'],
                PREPROC['augment']['crop'],
                PREPROC['augment']['flip']
            )
        )


def build_model(kernel=5, l2=5e-4, dropout=0.2, lr=0.01, stride=1, stats=None):
    reg = regularizers.l2(l2) if l2 > 0 else None
    he  = initializers.HeNormal()

    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Rescaling(1./255)(inputs)

    if PREPROC["per_channel_standardize"] and stats is not None:
        norm_layer = layers.Normalization(
            mean=stats["mean"],
            variance=[s*s for s in stats["std"]],
            name="in_norm"
        )
        x = norm_layer(x)

    x = augment()(x)

    # 小型 VGG-style
    x = layers.Conv2D(64, kernel, strides=stride, padding="same", use_bias=False,
                      kernel_initializer=he, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(64, kernel, padding="same", use_bias=False,
                      kernel_initializer=he, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(128, kernel, padding="same", use_bias=False,
                      kernel_initializer=he, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(128, kernel, padding="same", use_bias=False,
                      kernel_initializer=he, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(256, kernel, padding="same", use_bias=False,
                      kernel_initializer=he, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer=he, kernel_regularizer=reg)(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    # Cosine 衰減（整個訓練期衰減一次）
    steps_per_epoch = 45000 // BATCH
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=steps_per_epoch * EPOCHS
    )

    # SGD + Nesterov
    opt = optimizers.SGD(learning_rate=lr_sched, momentum=0.9, nesterov=True)

    model = tf.keras.Model(inputs, outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model


def run_one(exp):
    name = exp["name"]
    out_dir = os.path.join(OUT_DIR, f'{time_stamp()}_{name}')
    ensure_dir(out_dir)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten(); y_test = y_test.flatten()

    #preprocess the data
    x_train, x_test, stats, x0_train, x0_test = preprocess_data(x_train, x_test)

    # 建立模型與增強層（增強只用在訓練圖）
    model = build_model(kernel=exp["kernel"], l2=exp["l2"],
                    dropout=exp["dropout"], lr=exp["lr"],
                    stride=exp.get("stride", 1), stats=stats)
    # 生成前處理報告與圖檔
    save_preproc_report(os.path.join(out_dir, "preprocess"), x0_train, x_train, stats, augment())

    print(model.summary())
    cbs = [
        callbacks.ModelCheckpoint(os.path.join(out_dir, "best.keras"),
                                monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=15,  
                                restore_best_weights=True, verbose=1),
    ]
    test_cb = TestEvalCallback(x_test, y_test); cbs.append(test_cb)

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cbs,
        verbose=2
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    save_history(history, os.path.join(out_dir, "history.json"))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Params: {param_count(model)}\n")
        f.write(f"Test acc: {test_acc:.4f}\n")
        f.write(f"Config: {exp}\n")
        f.write("Preprocess: scale01={}, per-channel standardize={}, "
                "aug=pad-{}/crop-{}/flip-{}\n".format(
                    PREPROC["scale01"], PREPROC["per_channel_standardize"],
                    PREPROC["augment"]["pad"], PREPROC["augment"]["crop"], PREPROC["augment"]["flip"])
               )

    # === 繪圖與展示===
    plot_learning_curves(history, os.path.join(out_dir, "learning.png"))
    plot_weight_bias_hists(model, os.path.join(out_dir, "hists_all"))
    plot_weight_bias_hists(model, os.path.join(out_dir, "hists_dense"), include={"Dense"})

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    plot_examples(x_test, y_test, y_pred, class_names, os.path.join(out_dir, "examples_correct.png"), correct=True)
    plot_examples(x_test, y_test, y_pred, class_names, os.path.join(out_dir, "examples_wrong.png"),   correct=False)

    idx = int(np.random.randint(0, len(x_test)))
    conv_names = [layer.name for layer in model.layers if "conv2d" in layer.name][:2]
    if conv_names:
        plot_feature_maps(model, conv_names, x_test[idx], os.path.join(out_dir, "feature_maps"))

    from common.plotting import make_dashboard
    make_dashboard(history, {"acc": test_cb.test_acc, "loss": test_cb.test_loss},
                   model, os.path.join(out_dir, "dashboard.png"), bins=100)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for exp in EXPERIMENTS:
        run_one(exp)
