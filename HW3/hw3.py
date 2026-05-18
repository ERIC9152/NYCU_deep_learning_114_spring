
# -*- coding: utf-8 -*-
import os, io, math, argparse, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =============== 工具函式 ===============
def load_text(train_path, valid_path=None, encoding="utf8"):
    with io.open(train_path, "r", encoding=encoding) as f:
        train_text = f.read()
    if valid_path and os.path.exists(valid_path):
        with io.open(valid_path, "r", encoding=encoding) as f:
            valid_text = f.read()
    else:
        # 若無 valid，從 train 後半段切一份
        cut = len(train_text) // 2
        valid_text = train_text[cut:]
        train_text = train_text[:cut]
    return train_text, valid_text

def build_vocab(train_text):
    vocab = sorted(list(set(train_text)))
    v2i = {c:i for i,c in enumerate(vocab)}
    i2v = {i:c for c,i in v2i.items()}
    return vocab, v2i, i2v

def encode(text, v2i):
    unk = 0
    return np.array([v2i.get(c, unk) for c in text], dtype=np.int32)

def make_dataset(ids, seq_len, batch, buffer_size=100000):
    # 將連續字元 id 切成長度 seq_len+1 的片段，再分割成 (x[:seq_len], x[1:])
    ds = tf.data.Dataset.from_tensor_slices(ids)
    ds = ds.batch(seq_len + 1, drop_remainder=True)
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    ds = ds.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size).batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# =============== 模型 ===============
def make_model(vocab_size, embed_dim, hidden, model_type="lstm",
               layers=1, dropout=0.1, recurrent_dropout=0.0):
    RNN = {"rnn": tf.keras.layers.SimpleRNN,
           "lstm": tf.keras.layers.LSTM,
           "gru": tf.keras.layers.GRU}[model_type]
    inp = tf.keras.Input(shape=(None,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, name="embed")(inp)
    for li in range(layers):
        x = RNN(hidden, return_sequences=True,
                dropout=dropout, recurrent_dropout=recurrent_dropout,
                name=f"{model_type}_{li+1}")(x)
    logits = tf.keras.layers.Dense(vocab_size, name="lm_head")(x)
    return tf.keras.Model(inp, logits)

# =============== 文字生成 ===============
def generate_text(model, i2v, v2i, prime="JULIET", gen_len=300, temperature=1.0, seq_len=80):
    vocab_size = len(i2v)
    out = list(prime)
    def sample_from_logits(logits):
        if temperature <= 0:
            return int(tf.argmax(logits, axis=-1).numpy())
        logits = logits / float(temperature)
        probs = tf.nn.softmax(logits).numpy()
        return int(np.random.choice(vocab_size, p=probs))
    for _ in range(gen_len):
        recent = out[-seq_len:]
        token_ids = np.array([[v2i.get(c, 0) for c in recent]], dtype=np.int32)  # [1, T]
        logits = model(token_ids, training=False).numpy()        # [1, T, V]
        last_logits = logits[0, -1]                              # [V]
        nxt = sample_from_logits(last_logits)
        out.append(i2v[nxt])
    return "".join(out)

# =============== 回呼與繪圖 ===============
class BPCLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self.logs_all = []

    @staticmethod
    def _to_bpc(loss):
        return float(loss) / math.log(2.0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        rec = {
            "epoch": int(epoch + 1),
            "loss": float(logs.get("loss", float("nan"))),
            "val_loss": float(logs.get("val_loss", float("nan"))) if "val_loss" in logs else None,
        }
        rec["bpc"] = self._to_bpc(rec["loss"])
        rec["val_bpc"] = self._to_bpc(rec["val_loss"]) if rec["val_loss"] is not None else None
        self.logs_all.append(rec)
        with io.open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.logs_all, f, ensure_ascii=False, indent=2)

# 【新增】固定斷點儲存 + 生成
class BreakpointSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, break_epochs, i2v, v2i, prime, gen_len, temperature, seq_len):
        super().__init__()
        self.save_dir = save_dir
        self.break_epochs = set(int(e) for e in break_epochs)
        self.i2v = i2v
        self.v2i = v2i
        self.prime = prime
        self.gen_len = gen_len
        self.temperature = temperature
        self.seq_len = seq_len

    def on_epoch_end(self, epoch, logs=None):
        ep = int(epoch + 1)
        if ep in self.break_epochs:
            # 存權重
            wpath = os.path.join(self.save_dir, f"snap_ep{ep:02d}.weights.h5")
            self.model.save_weights(wpath)
            # 立即生成一段文字（對應該 epoch 的模型）
            txt = generate_text(self.model, self.i2v, self.v2i,
                                prime=self.prime, gen_len=self.gen_len,
                                temperature=self.temperature, seq_len=self.seq_len)
            tpath = os.path.join(self.save_dir, f"sample_ep{ep:02d}.txt")
            with io.open(tpath, "w", encoding="utf-8") as f:
                f.write(txt)

def plot_curves(history, save_dir, title_prefix="", break_epochs=None):
    import math, numpy as np, os, matplotlib.pyplot as plt
    break_epochs = sorted(set(int(e) for e in (break_epochs or [])))

    # === (A) BPC ===
    tr = np.array(history.history.get("loss", []), dtype=np.float64)
    vl = np.array(history.history.get("val_loss", []), dtype=np.float64) if "val_loss" in history.history else None
    ep = np.arange(1, len(tr) + 1)  # 1-based epoch index
    tr_bpc = tr / math.log(2.0)
    plt.figure()
    plt.plot(ep, tr_bpc, label="train BPC")
    if vl is not None:
        plt.plot(ep, vl / math.log(2.0), label="val BPC")
    # 在 BPC 圖上也可選擇標記（需要就取消註解）
    # for bp in break_epochs:
    #     if 1 <= bp <= len(ep):
    #         plt.axvline(bp, linestyle="--", linewidth=0.8)
    #         plt.scatter(bp, tr_bpc[bp-1], zorder=3)
    #         plt.text(bp, tr_bpc[bp-1], f"ep{bp:02d}", fontsize=8, ha="left", va="bottom")
    plt.title(f"{title_prefix}BPC per epoch")
    plt.xlabel("Epoch"); plt.ylabel("Bits per character"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bpc_curve.png")); plt.close()

    # === (B) Accuracy & Error rate ===
    acc = history.history.get("sparse_categorical_accuracy")
    val_acc = history.history.get("val_sparse_categorical_accuracy")

    if acc is not None:
        acc = np.asarray(acc, dtype=np.float64)
        val_acc = np.asarray(val_acc, dtype=np.float64) if val_acc is not None else None

        # Accuracy
        plt.figure()
        plt.plot(ep, acc, label="train acc")
        if val_acc is not None:
            plt.plot(ep, val_acc, label="val acc")
        plt.title(f"{title_prefix}Accuracy per epoch")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "acc_curve.png")); plt.close()

        # Error rate = 1 - Accuracy（這張圖上「畫出 Breakpoints」）
        tr_err = 1.0 - acc
        vl_err = 1.0 - val_acc if val_acc is not None else None
        plt.figure()
        plt.plot(ep, tr_err, label="train error rate")
        if vl_err is not None:
            plt.plot(ep, vl_err, label="val error rate")

        # 在 3/6/9/12/15 做標記（虛線 + 圓點 + 小標籤）
        for bp in break_epochs:
            if 1 <= bp <= len(ep):
                plt.axvline(bp, linestyle="--", linewidth=0.8)
                # 標註訓練誤差點
                plt.scatter(bp, tr_err[bp-1], zorder=3)
                plt.text(bp, tr_err[bp-1], f"ep{bp:02d}", fontsize=8, ha="left", va="bottom")
                # 若有驗證誤差，也一併標
                if vl_err is not None:
                    plt.scatter(bp, vl_err[bp-1], zorder=3)
                    plt.text(bp, vl_err[bp-1], f"ep{bp:02d}", fontsize=8, ha="right", va="top")

        plt.title(f"{title_prefix}Error rate per epoch")
        plt.xlabel("Epoch"); plt.ylabel("Error rate"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "error_curve.png")); plt.close()
# =============== 主流程 ===============
def main():
    ap = argparse.ArgumentParser(description="TensorFlow Keras Char-level LM")
    ap.add_argument("--model", choices=["rnn","lstm","gru"], default="lstm")
    ap.add_argument("--data-train", default="shakespeare_train.txt")
    ap.add_argument("--data-valid", default="shakespeare_valid.txt")
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default="keras_outputs")
    ap.add_argument("--prime", default="JULIET")
    ap.add_argument("--gen-len", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--global-clipnorm", type=float, default=None, help="全域 L2 梯度剪裁上限（例：1.0）")
    ap.add_argument("--clipvalue", type=float, default=None, help="逐元素梯度剪裁上限（與 global-clipnorm 擇一）")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recurrent-dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    # 【新增】斷點 epoch，預設即 3,6,9,12,15（符合你指定且 epochs=15）
    ap.add_argument("--break-epochs", type=int, nargs="*", default=[3,6,9,12,15],
                    help="在這些 epochs 結束時保存快照並生成文字")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    # 資料與字典
    train_text, valid_text = load_text(args.data_train, args.data_valid)
    vocab, v2i, i2v = build_vocab(train_text)
    V = len(vocab)
    train_ids = encode(train_text, v2i)
    valid_ids = encode(valid_text, v2i)

    # tf.data pipeline
    ds_train = make_dataset(train_ids, args.seq_len, args.batch)
    ds_valid = make_dataset(valid_ids, args.seq_len, args.batch)

    # 模型建立與編譯
    model = make_model(V, args.embed, args.hidden, model_type=args.model,
                       layers=args.layers, dropout=args.dropout,
                       recurrent_dropout=args.recurrent_dropout)

    # ---- Optimizer----
    weight_decay = getattr(args, "weight_decay", 0.0)
    try:
        if weight_decay and weight_decay > 0:
            AdamW = getattr(tf.keras.optimizers, "AdamW",
                            getattr(tf.keras.optimizers.experimental, "AdamW", None))
            if AdamW is not None:
                opt = AdamW(
                    learning_rate=args.lr,
                    weight_decay=weight_decay,
                    global_clipnorm=args.global_clipnorm,
                    clipvalue=args.clipvalue
                )
            else:
                opt = tf.keras.optimizers.Adam(
                    learning_rate=args.lr,
                    global_clipnorm=args.global_clipnorm,
                    clipvalue=args.clipvalue
                )
        else:
            opt = tf.keras.optimizers.Adam(
                learning_rate=args.lr,
                global_clipnorm=args.global_clipnorm,
                clipvalue=args.clipvalue
            )
    except Exception:
        opt = tf.keras.optimizers.Adam(
            learning_rate=args.lr,
            global_clipnorm=args.global_clipnorm,
            clipvalue=args.clipvalue
        )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss_fn,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # ---- End Optimizer ----

    # 訓練（含斷點與最佳保存）
    csv_log = tf.keras.callbacks.CSVLogger(os.path.join(args.save_dir, "history.csv"))
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.save_dir, "weights.weights.h5"),
        save_weights_only=True, save_best_only=True,
        monitor="val_loss", mode="min"
    )
    bpc_log = BPCLogger(os.path.join(args.save_dir, "bpc_log.json"))

    # 固定斷點（3,6,9,12,15）
    bp = BreakpointSaver(
        save_dir=args.save_dir,
        break_epochs=args.break_epochs,
        i2v=i2v, v2i=v2i,
        prime=args.prime, gen_len=args.gen_len,
        temperature=args.temperature, seq_len=args.seq_len
    )

    history = model.fit(
        ds_train, validation_data=ds_valid,
        epochs=args.epochs,
        callbacks=[csv_log, ckpt, bpc_log, bp]
    )

    # 繪圖
    plot_curves(history, args.save_dir, title_prefix=f"{args.model.upper()} ",
            break_epochs=args.break_epochs)

    # 斷點文字生成（使用最佳權重再產一份總結 sample）
    model.load_weights(os.path.join(args.save_dir, "weights.weights.h5"))
    sample = generate_text(model, i2v, v2i,
                           prime=args.prime, gen_len=args.gen_len,
                           temperature=args.temperature, seq_len=args.seq_len)
    with io.open(os.path.join(args.save_dir, f"sample_{args.model}.txt"), "w", encoding="utf-8") as f:
        f.write(sample)

    # 存字典與設定
    with io.open(os.path.join(args.save_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f, ensure_ascii=False, indent=2)
    with io.open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # 總結 README
    with io.open(os.path.join(args.save_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# CharLM ({args.model.upper()})\n")
        f.write(f"- vocab size: {V}\n- embed: {args.embed}, hidden: {args.hidden}, seq_len: {args.seq_len}, batch: {args.batch}, epochs: {args.epochs}, lr: {args.lr}\n")
        f.write(f"- samples: sample_{args.model}.txt + sample_ep03/06/09/12/15.txt\n")
        f.write(f"- curves: bpc_curve.png, acc_curve.png, error_curve.png\n")
    print(f"完成！輸出路徑：{args.save_dir}")
    print("檔案：weights.weights.h5, snap_ep*.weights.h5, sample_*.txt, bpc_curve.png, history.csv, vocab.json, config.json, README.md")

if __name__ == "__main__":
    main()