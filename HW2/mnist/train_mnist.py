
import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, optimizers, initializers
from common.utils import set_seed, ensure_dir, param_count, save_history, time_stamp
from common.plotting import (plot_learning_curves, plot_weight_bias_hists, plot_examples, plot_feature_maps, make_dashboard)

set_seed(1337)

OUT_DIR = "outputs/mnist"
EPOCHS = 15
BATCH = 128

EXPERIMENTS = []
for k in (3, 5):
    for s in (1, 2):
        for l2 in (0.0, 1e-4):
            for dr in (0.0, 0.3):
                EXPERIMENTS.append({
                    "name": f"k{k}_s{s}_l2{l2}_dr{dr}",
                    "kernel": k, "stride": s, "l2": l2, "dropout": dr
                })

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

num_classes = 10
class_names = [str(i) for i in range(num_classes)]

class TestEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test, self.y_test = x_test, y_test
        self.test_loss, self.test_acc = [], []
    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.test_loss.append(float(loss))
        self.test_acc.append(float(acc))

def build_model(kernel=3, stride=1, l2=0.0, dropout=0.3):
    reg = regularizers.l2(l2) if l2>0 else None
    he = initializers.HeNormal()

    inputs = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, kernel_size=kernel, strides=stride, padding="same",
                      activation=None, use_bias=False, kernel_initializer=he,
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(64, kernel_size=kernel, strides=1, padding="same",
                      activation=None, use_bias=False, kernel_initializer=he,
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer=he,
                     kernel_regularizer=reg)(x)
    if dropout and dropout>0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform")(x)

    opt = optimizers.AdamW(learning_rate=1e-3, weight_decay=l2 if l2>0 else 0.0)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def run_one(exp):
    name = exp["name"]
    out_dir = os.path.join(OUT_DIR, f'{time_stamp()}_{name}')
    ensure_dir(out_dir)

    model = build_model(kernel=exp["kernel"], stride=exp["stride"], l2=exp["l2"], dropout=exp["dropout"])
    print(model.summary())

    cbs = [
        callbacks.ModelCheckpoint(os.path.join(out_dir, "best.keras"),
                                  monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                    min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]
    test_cb = TestEvalCallback(x_test, y_test)   # ← 新增
    cbs.append(test_cb)                          # ← 新增

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

    plot_learning_curves(history, os.path.join(out_dir, "learning.png"))
    plot_weight_bias_hists(model, os.path.join(out_dir, "hists_all"))
    plot_weight_bias_hists(model, os.path.join(out_dir, "hists_dense"), include={"Dense"})

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    plot_examples(x_test, y_test, y_pred, class_names, os.path.join(out_dir, "examples.png"), correct=True)
    plot_examples(x_test, y_test, y_pred, class_names, os.path.join(out_dir, "examples.png"), correct=False)

    idx = int(np.random.randint(0, len(x_test)))
    conv_names = [layer.name for layer in model.layers if "conv2d" in layer.name][:2]
    if conv_names:
        plot_feature_maps(model, conv_names, x_test[idx], os.path.join(out_dir, "feature_maps"))
    make_dashboard(
    history,
    {"acc": test_cb.test_acc, "loss": test_cb.test_loss},
    model,
    os.path.join(out_dir, "dashboard.png"), bins=100)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for exp in EXPERIMENTS:
        run_one(exp)
