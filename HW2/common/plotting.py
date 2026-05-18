
import os, numpy as np, matplotlib.pyplot as plt

def plot_learning_curves(history, out_png):
    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_loss.png"), dpi=140); plt.close()

    plt.figure()
    train_acc_key = "accuracy" if "accuracy" in history.history else "sparse_categorical_accuracy"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_sparse_categorical_accuracy"
    plt.plot(epochs, history.history.get(train_acc_key, []), label="train")
    if val_acc_key in history.history:
        plt.plot(epochs, history.history[val_acc_key], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_acc.png"), dpi=140); plt.close()

def plot_weight_bias_hists(model, out_dir, include=None, exclude=None, bins=50):
    os.makedirs(out_dir, exist_ok=True)
    for i, layer in enumerate(model.layers):
        cls = layer.__class__.__name__
        if include and cls not in include:
            continue
        if exclude and cls in exclude:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        for j, w in enumerate(weights):
            plt.figure()
            plt.hist(np.ravel(w), bins=bins)
            plt.title(f"{i:02d}:{layer.name} ({cls})  w{j}")
            plt.tight_layout()
            fname = os.path.join(out_dir, f"{i:02d}_{layer.name}_{cls}_w{j}_hist.png")
            plt.savefig(fname, dpi=140)
            plt.close()

def plot_examples(x, y_true, y_pred, class_names, out_png, correct=True, max_examples=25):
    idxs = np.where((y_true == y_pred) if correct else (y_true != y_pred))[0]
    if len(idxs) == 0:
        return
    idxs = idxs[:max_examples]
    side = int(np.ceil(np.sqrt(len(idxs))))
    plt.figure(figsize=(side*2, side*2))
    for k, idx in enumerate(idxs):
        plt.subplot(side, side, k+1)
        img = x[idx]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
            plt.imshow(img, cmap="gray")
        elif img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
        plt.title(f"T:{class_names[int(y_true[idx])]} P:{class_names[int(y_pred[idx])]}")
    plt.tight_layout()
    tag = "correct" if correct else "misclassified"
    plt.savefig(out_png.replace(".png", f"_{tag}.png"), dpi=160)
    plt.close()

def plot_feature_maps(model, layer_names, img, out_dir):
    import tensorflow as tf
    os.makedirs(out_dir, exist_ok=True)
    outputs = [model.get_layer(name).output for name in layer_names]
    fm_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    fmaps = fm_model.predict(img[None, ...], verbose=0)
    for name, fmap in zip(layer_names, fmaps):
        fmap = np.array(fmap[0])
        num = min(16, fmap.shape[-1])
        side = int(np.ceil(np.sqrt(num)))
        plt.figure(figsize=(side*2, side*2))
        for i in range(num):
            plt.subplot(side, side, i+1)
            plt.imshow(fmap[..., i], cmap="viridis")
            plt.axis("off"); plt.title(f"{name} #{i}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_tiles.png"), dpi=160)
        plt.close()

def _hist_on_ax(ax, arr, title, bins=100):
    ax.hist(np.ravel(arr), bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Number")

def make_dashboard(history, test_metrics, model, out_png, bins=100):
    # 取 key
    train_acc_key = "accuracy" if "accuracy" in history.history else "sparse_categorical_accuracy"
    val_acc_key   = "val_accuracy" if "val_accuracy" in history.history else "val_sparse_categorical_accuracy"

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # (1) Accuracy：train / val / test
    axs[0,0].plot(epochs, history.history.get(train_acc_key, []), label="Training Accuracy")
    if val_acc_key in history.history:
        axs[0,0].plot(epochs, history.history[val_acc_key], label="Validation Accuracy")
    if test_metrics and "acc" in test_metrics:
        axs[0,0].plot(epochs, test_metrics["acc"], label="Test Accuracy")
    axs[0,0].set_title("Accuracy")
    axs[0,0].set_xlabel("Iteration"); axs[0,0].set_ylabel("Accuracy rate")
    axs[0,0].legend(loc="lower right")

    # (2) Learning Curve（Loss）
    axs[0,1].plot(epochs, history.history["loss"], label="Cross entropy")
    if "val_loss" in history.history:
        axs[0,1].plot(epochs, history.history["val_loss"], label="Val loss")
    axs[0,1].set_title("Learning Curve")
    axs[0,1].set_xlabel("Iteration"); axs[0,1].set_ylabel("Loss")
    axs[0,1].legend(loc="upper right")

    # 選四個層：兩個 Conv + 一個 Dense（非輸出）+ 輸出 Dense
    conv_names  = [l.name for l in model.layers if l.__class__.__name__.lower().startswith("conv")][:2]
    dense_names = [l.name for l in model.layers if l.__class__.__name__ == "Dense"]
    if len(dense_names) >= 2:
        dense1_name = dense_names[-2]      # 倒數第二個：非輸出 dense
        output_name = dense_names[-1]      # 倒數第一個：輸出層
    elif len(dense_names) == 1:
        dense1_name = dense_names[0]
        output_name = dense_names[0]
    else:
        dense1_name = output_name = None

    # (3)(4)(5)(6) 權重直方圖（只畫 kernel，想畫 bias 改成 weights[1]）
    panels = [(1,0,"Histogram of conv1", conv_names[0] if len(conv_names)>0 else None),
              (1,1,"Histogram of conv2", conv_names[1] if len(conv_names)>1 else None),
              (1,2,"Histogram of dense1", dense1_name),
             ]
    # 把右上空一格（0,2）補成輸出層直方圖，佈局就跟你圖一樣三列兩欄
    panels.insert(0, (0,2,"Histogram of output", output_name))

    for (r,c,title,layer_name) in panels:
        ax = axs[r,c]
        if layer_name is None:
            ax.axis("off"); ax.set_title(f"{title} (N/A)")
            continue
        w = model.get_layer(layer_name).get_weights()
        if not w:  # 例如 BN
            ax.axis("off"); ax.set_title(f"{title} (no weights)")
            continue
        _hist_on_ax(ax, w[0], title, bins=bins)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()