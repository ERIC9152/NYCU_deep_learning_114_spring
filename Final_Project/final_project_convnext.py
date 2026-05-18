# ==================================================
# final_project_convnext_resblock_mlp_freeze.py
# Architecture (as requested):
# - Backbone: ConvNeXt (frozen during training)
# - Feature extractor: Residual Blocks (trainable)
# - Classifier: MLP
# Keras 3 compatible, jit disabled
# ==================================================

import os

# =========================
# MUST set before importing TF
# =========================
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

# =========================
# Config
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# !!! 確認路徑 !!!
train_dir = "/mnt/d/DL/Final_Project/train"
test_dir  = "/mnt/d/DL/Final_Project/test"

# GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# Mixed precision (optional)
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass


# =========================
# Extra augmentations (TRAIN) - for better generalization
# =========================
def random_resize_jitter(img, min_scale=0.75, max_scale=1.15):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.random.uniform([], min_scale, max_scale)
    nh = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    nw = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img2 = tf.image.resize(img, [nh, nw], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, h, w)
    return img2

def random_gaussian_noise(img, sigma_min=0.0, sigma_max=6.0):
    sigma = tf.random.uniform([], sigma_min, sigma_max)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=sigma, dtype=img.dtype)
    return tf.clip_by_value(img + noise, 0.0, 255.0)

def random_blur(img, k=3):
    do = tf.random.uniform([]) < 0.35
    def _blur(x):
        x4 = tf.expand_dims(x, 0)
        x4 = tf.nn.avg_pool2d(x4, ksize=k, strides=1, padding="SAME")
        return tf.squeeze(x4, 0)
    return tf.cond(do, lambda: _blur(img), lambda: img)

def random_cutout(img, max_frac=0.25):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    cut_h = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform([], 0.05, max_frac), tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform([], 0.05, max_frac), tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)

    mask_y = tf.logical_and(tf.range(h)[:, None] >= y1, tf.range(h)[:, None] < y2)
    mask_x = tf.logical_and(tf.range(w)[None, :] >= x1, tf.range(w)[None, :] < x2)
    mask = tf.cast(tf.logical_and(mask_y, mask_x), img.dtype)
    mask = tf.expand_dims(mask, -1)  # [H,W,1]
    return img * (1.0 - mask)

def _jpeg_recompress_py(img_u8, min_q, max_q):
    import numpy as _np
    import tensorflow as _tf
    q = _np.random.randint(int(min_q), int(max_q) + 1)
    encoded = _tf.io.encode_jpeg(
        _tf.convert_to_tensor(img_u8, dtype=_tf.uint8),
        quality=int(q),
        chroma_downsampling=True
    )
    decoded = _tf.io.decode_jpeg(encoded, channels=3)
    return decoded.numpy()

def random_jpeg_recompress(img_f32, min_q=60, max_q=100):
    img_u8 = tf.cast(tf.clip_by_value(img_f32, 0.0, 255.0), tf.uint8)
    out = tf.py_function(
        func=_jpeg_recompress_py,
        inp=[img_u8, tf.constant(min_q), tf.constant(max_q)],
        Tout=tf.uint8
    )
    out.set_shape(img_u8.shape)
    return tf.cast(out, tf.float32)

def augment_batch(images, labels, training=True):
    if not training:
        return images, labels

    # JPEG recompress (generator artifact robustness)
    images = tf.map_fn(
        lambda im: random_jpeg_recompress(im, 60, 100),
        images,
        fn_output_signature=tf.float32
    )

    # Resampling / blur / noise / cutout
    images = tf.map_fn(lambda im: random_resize_jitter(im, 0.75, 1.15), images, fn_output_signature=tf.float32)
    images = tf.map_fn(lambda im: random_blur(im, k=3), images, fn_output_signature=tf.float32)
    images = tf.map_fn(lambda im: random_gaussian_noise(im, 0.0, 6.0), images, fn_output_signature=tf.float32)

    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, 0.10)
    images = tf.image.random_contrast(images, 0.85, 1.15)
    images = tf.image.random_saturation(images, 0.85, 1.20)
    images = tf.image.random_hue(images, 0.02)

    # Random erasing (cutout)
    do = tf.random.uniform([]) < 0.35
    images = tf.cond(
        do,
        lambda: tf.map_fn(lambda im: random_cutout(im, 0.25), images, fn_output_signature=tf.float32),
        lambda: images
    )

    return images, labels


# =========================
# Serializable Layers
# =========================
@tf.keras.utils.register_keras_serializable()
class ConvNeXtPreprocessLayer(layers.Layer):
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        x = convnext_preprocess(x)
        return tf.cast(x, inputs.dtype)

@tf.keras.utils.register_keras_serializable()
class ResidualBlock(layers.Layer):
    """Basic ResNet-style residual block (Conv-BN-ReLU-Conv-BN + skip)."""
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.stride = int(stride)

        self.c1 = layers.Conv2D(self.filters, 3, strides=self.stride, padding="same",
                                use_bias=False, kernel_initializer="he_normal")
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation("relu")

        self.c2 = layers.Conv2D(self.filters, 3, strides=1, padding="same",
                                use_bias=False, kernel_initializer="he_normal")
        self.b2 = layers.BatchNormalization()

        self.proj = None
        self.a2 = layers.Activation("relu")

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if self.stride != 1 or in_ch != self.filters:
            self.proj = models.Sequential([
                layers.Conv2D(self.filters, 1, strides=self.stride, padding="same",
                              use_bias=False, kernel_initializer="he_normal"),
                layers.BatchNormalization()
            ])
        super().build(input_shape)

    def call(self, x, training=None):
        shortcut = x

        y = self.c1(x)
        y = self.b1(y, training=training)
        y = self.a1(y)

        y = self.c2(y)
        y = self.b2(y, training=training)

        if self.proj is not None:
            shortcut = self.proj(shortcut, training=training)

        y = layers.add([y, shortcut])
        y = self.a2(y)
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "stride": self.stride})
        return cfg


# =========================
# Model
# =========================
def build_model():
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

    # Backbone: ConvNeXt (frozen during training)
    x = ConvNeXtPreprocessLayer(name="convnext_preprocess")(inp)
    backbone = ConvNeXtTiny(include_top=False, weights="imagenet", input_tensor=x)
    backbone.trainable = False  # <<< freeze backbone as requested

    feat = backbone.output  # feature map

    # Residual blocks as feature extractor (trainable)
    # 先用 1x1 降維 + BN + ReLU，再堆疊 residual blocks
    y = layers.Conv2D(512, 1, padding="same", use_bias=False, name="reduce_1x1")(feat)
    y = layers.BatchNormalization(name="reduce_bn")(y)
    y = layers.Activation("relu", name="reduce_relu")(y)

    # Stage 1
    y = ResidualBlock(512, stride=1, name="resblk1")(y)
    y = ResidualBlock(512, stride=1, name="resblk2")(y)

    # Stage 2 (downsample)
    y = ResidualBlock(256, stride=2, name="resblk3")(y)
    y = ResidualBlock(256, stride=1, name="resblk4")(y)

    # Stage 3 (downsample)
    y = ResidualBlock(128, stride=2, name="resblk5")(y)
    y = ResidualBlock(128, stride=1, name="resblk6")(y)

    # Pooling
    y = layers.GlobalAveragePooling2D(name="gap")(y)

    # MLP classifier
    y = layers.Dense(512, activation="relu", name="mlp_fc1")(y)
    y = layers.BatchNormalization(name="mlp_bn1")(y)
    y = layers.Dropout(0.5, name="mlp_dp1")(y)

    y = layers.Dense(128, activation="relu", name="mlp_fc2")(y)
    y = layers.BatchNormalization(name="mlp_bn2")(y)
    y = layers.Dropout(0.3, name="mlp_dp2")(y)

    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(y)

    return models.Model(inputs=inp, outputs=out, name="ConvNeXt_ResBlocks_MLP_FrozenBackbone")


# =========================
# Dataset
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
class_to_idx = {name: i for i, name in enumerate(class_names)}
print("class_names:", class_names)
print("class_to_idx:", class_to_idx)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.map(lambda x, y: augment_batch(x, y, training=True), num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: augment_batch(x, y, training=False), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)


# =========================
# Callbacks
# =========================
cbs = [
    ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, mode="min"),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
]


# =========================
# Train (Backbone always frozen)
# =========================
model = build_model()
model.compile(
    optimizer=optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=3e-4, decay_steps=2000
        ),
        weight_decay=1e-4
    ),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=False
)

print("\n=== Train: ConvNeXt frozen + ResidualBlocks feature extractor + MLP classifier ===")
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=cbs)


# =========================
# Load best model
# =========================
print("\n==================================================")
print("Loading best model...")
print("==================================================")

model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={
        "ConvNeXtPreprocessLayer": ConvNeXtPreprocessLayer,
        "ResidualBlock": ResidualBlock,
    }
)


# =========================
# TTA helpers (inference only) - 4 WAY
# =========================
def _jpeg_fixed_quality_py(img_u8, quality):
    import tensorflow as _tf
    encoded = _tf.io.encode_jpeg(
        _tf.convert_to_tensor(img_u8, dtype=_tf.uint8),
        quality=int(quality),
        chroma_downsampling=True
    )
    decoded = _tf.io.decode_jpeg(encoded, channels=3)
    return decoded.numpy()

def jpeg_fixed_quality(img_f32, quality=90):
    img_u8 = tf.cast(tf.clip_by_value(img_f32, 0.0, 255.0), tf.uint8)
    out = tf.py_function(
        func=_jpeg_fixed_quality_py,
        inp=[img_u8, tf.constant(int(quality), dtype=tf.int32)],
        Tout=tf.uint8
    )
    out.set_shape(img_u8.shape)
    return tf.cast(out, tf.float32)

def predict_model_output_tta4(model, img_f32):
    a0 = img_f32
    a1 = tf.image.flip_left_right(img_f32)

    img_j90 = jpeg_fixed_quality(img_f32, 90)
    a2 = img_j90
    a3 = tf.image.flip_left_right(img_j90)

    batch = tf.stack([a0, a1, a2, a3], axis=0)  # [4,H,W,3]
    preds = model.predict(batch, verbose=0).reshape(-1)
    return float(np.mean(preds))


# =========================
# Predict & submission
# =========================
def load_test_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return tf.cast(arr, tf.float32)

test_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
    key=lambda x: int(os.path.splitext(x)[0])
)

if "fake" not in class_to_idx:
    raise RuntimeError(f"找不到 'fake' 類別！目前 class_names={class_names}")

fake_index = class_to_idx["fake"]
print("fake_index:", fake_index)

rows = []
for f in test_files:
    fp = os.path.join(test_dir, f)
    img = load_test_image(fp)

    p = predict_model_output_tta4(model, img)

    fake_prob = p if fake_index == 1 else (1.0 - p)
    pred_label = "fake" if fake_prob >= 0.5 else "real"
    rows.append([os.path.splitext(f)[0], pred_label])

sub = pd.DataFrame(rows, columns=["filename", "label"])
sub.to_csv("convnext_resmlp_submission.csv", index=False)
print("Saved: convnext_resmlp_submission.csv")
print(sub.head())
