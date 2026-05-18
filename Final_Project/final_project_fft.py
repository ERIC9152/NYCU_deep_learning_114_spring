# ==================================================
# final_project_fft_batch16_nojit_tta4.py (Fixed for Keras 3)
# - 修正內容：將 tf.stop_gradient 封裝進自定義 Layer
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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

# Mixed precision
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass



# =========================
# Extra augmentations (TRAIN) - for better generalization to unseen generators
# - resize jitter (simulate resampling)
# - gaussian noise / slight blur
# - cutout (random erasing)
# =========================
def random_resize_jitter(img, min_scale=0.75, max_scale=1.15):
    # img: float32 [H,W,3] in [0,255]
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.random.uniform([], min_scale, max_scale)
    nh = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    nw = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    # resize to random size then center crop/pad back
    img2 = tf.image.resize(img, [nh, nw], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, h, w)
    return img2

def random_gaussian_noise(img, sigma_min=0.0, sigma_max=6.0):
    sigma = tf.random.uniform([], sigma_min, sigma_max)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=sigma, dtype=img.dtype)
    return tf.clip_by_value(img + noise, 0.0, 255.0)

def random_blur(img, k=3):
    # cheap separable blur via average pool
    do = tf.random.uniform([]) < 0.35
    def _blur(x):
        x4 = tf.expand_dims(x, 0)
        x4 = tf.nn.avg_pool2d(x4, ksize=k, strides=1, padding="SAME")
        return tf.squeeze(x4, 0)
    return tf.cond(do, lambda: _blur(img), lambda: img)

def random_cutout(img, max_frac=0.25):
    # fill a random rectangle with 0 (black). Keeps dtype/shape.
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

# =========================
# JPEG recompress augmentation (TRAIN)
# =========================
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

    # JPEG recompress (stronger signal for generator artifacts)
    images = tf.map_fn(
        lambda im: random_jpeg_recompress(im, 60, 100),
        images,
        fn_output_signature=tf.float32
    )

    # Resampling / blur / noise / cutout: improve generalization to unseen generators
    images = tf.map_fn(lambda im: random_resize_jitter(im, 0.75, 1.15), images, fn_output_signature=tf.float32)
    images = tf.map_fn(lambda im: random_blur(im, k=3), images, fn_output_signature=tf.float32)
    images = tf.map_fn(lambda im: random_gaussian_noise(im, 0.0, 6.0), images, fn_output_signature=tf.float32)

    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, 0.10)
    images = tf.image.random_contrast(images, 0.85, 1.15)
    images = tf.image.random_saturation(images, 0.85, 1.20)
    images = tf.image.random_hue(images, 0.02)

    # Random erasing (cutout) with probability
    do = tf.random.uniform([]) < 0.35
    images = tf.cond(
        do,
        lambda: tf.map_fn(lambda im: random_cutout(im, 0.25), images, fn_output_signature=tf.float32),
        lambda: images
    )

    return images, labels

    images = tf.map_fn(
        lambda im: random_jpeg_recompress(im, 60, 100),
        images,
        fn_output_signature=tf.float32
    )
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, 0.08)
    images = tf.image.random_contrast(images, 0.90, 1.10)
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
class FFTAnalysisLayer(layers.Layer):
    def call(self, inputs):
        inputs_f32 = tf.cast(inputs, tf.float32)
        gray = tf.image.rgb_to_grayscale(inputs_f32)
        gray = tf.squeeze(gray, axis=-1)

        fft = tf.signal.fft2d(tf.cast(gray, tf.complex64))
        fft_shift = tf.signal.fftshift(fft)
        mag = tf.math.log(tf.abs(fft_shift) + 1.0)

        mean = tf.reduce_mean(mag, axis=[1, 2], keepdims=True)
        std  = tf.math.reduce_std(mag, axis=[1, 2], keepdims=True) + 1e-6
        mag = (mag - mean) / std

        mag = tf.expand_dims(mag, axis=-1)
        return tf.cast(mag, inputs.dtype)

@tf.keras.utils.register_keras_serializable()
class ResidualHighPassLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        k = np.array([[0, -1,  0],
                      [-1,  4, -1],
                      [0, -1,  0]], dtype=np.float32)
        self.kernel = tf.constant(k.reshape(3, 3, 1, 1), dtype=tf.float32)

    def call(self, inputs):
        gray = tf.image.rgb_to_grayscale(inputs)
        kernel = tf.cast(self.kernel, gray.dtype)
        hp = tf.nn.conv2d(gray, kernel, strides=1, padding="SAME")

        mean = tf.reduce_mean(hp, axis=[1, 2, 3], keepdims=True)
        std  = tf.math.reduce_std(hp, axis=[1, 2, 3], keepdims=True) + 1e-6
        hp = (hp - mean) / std
        return hp

@tf.keras.utils.register_keras_serializable()
class StopGradientLayer(layers.Layer):
    """修正 Keras 3 不能直接在 Functional API 中使用 tf.stop_gradient 的問題"""
    def call(self, inputs):
        return tf.stop_gradient(inputs)


# =========================
# Model
# =========================
def build_model():
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

    # RGB backbone
    x_rgb_in = ConvNeXtPreprocessLayer(name="convnext_preprocess")(inp)
    backbone = ConvNeXtTiny(include_top=False, weights="imagenet", input_tensor=x_rgb_in)
    backbone.trainable = False

    x_rgb = layers.GlobalAveragePooling2D(name="rgb_gap")(backbone.output)
    x_rgb = layers.Dropout(0.3, name="rgb_dropout")(x_rgb)

    # FFT branch
    x_fft = FFTAnalysisLayer(name="fft_layer")(inp)
    x_fft = StopGradientLayer(name="fft_stop_grad")(x_fft) # 使用自定義 Layer 修正
    x_fft = layers.Conv2D(32, 3, padding="same", activation="relu", name="fft_c1")(x_fft)
    x_fft = layers.MaxPooling2D(name="fft_pool")(x_fft)
    x_fft = layers.Conv2D(64, 3, padding="same", activation="relu", name="fft_c2")(x_fft)
    x_fft = layers.GlobalAveragePooling2D(name="fft_gap")(x_fft)

    # Residual branch
    x_res = ResidualHighPassLayer(name="residual_high_pass_layer")(inp)
    x_res = StopGradientLayer(name="res_stop_grad")(x_res) # 使用自定義 Layer 修正
    x_res = layers.Conv2D(32, 3, padding="same", activation="relu", name="res_c1")(x_res)
    x_res = layers.MaxPooling2D(name="res_pool")(x_res)
    x_res = layers.Conv2D(64, 3, padding="same", activation="relu", name="res_c2")(x_res)
    x_res = layers.GlobalAveragePooling2D(name="res_gap")(x_res)

    x = layers.Concatenate(name="fusion")([x_rgb, x_fft, x_res])
    x = layers.Dense(256, activation="relu", name="head_fc")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)

    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)
    return models.Model(inputs=inp, outputs=out, name="RGB_FFT_RES_ConvNeXt")


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
    ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, mode="min"),    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
]


# =========================
# Phase 1
# =========================
model = build_model()
model.compile(
    # TF 的 CosineDecay 參數名是 initial_learning_rate（不是 learning_rate）
    optimizer=optimizers.AdamW(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=3e-4,
            decay_steps=2000,
        ),
        weight_decay=1e-4,
    ),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=False
)

print("\n=== Phase 1: Train head (backbone frozen) ===")
model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=cbs)


# =========================
# Phase 2 (fine-tune only last 10% ConvNeXt)
# =========================
backbone = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "convnext" in layer.name.lower():
        backbone = layer
        break
if backbone is None:
    raise RuntimeError("找不到 ConvNeXt backbone，請用 model.summary() 檢查。")

backbone.trainable = True
n = len(backbone.layers)

for l in backbone.layers[: int(0.7 * n)]:
    l.trainable = False

model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-5, weight_decay=5e-5),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=False
)

print("\n=== Phase 2: Fine-tune last 30% blocks ===")
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=cbs)


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
        "FFTAnalysisLayer": FFTAnalysisLayer,
        "ResidualHighPassLayer": ResidualHighPassLayer,
        "StopGradientLayer": StopGradientLayer,
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
    # Build a batch of 4 augmented views and predict once (faster than 4x predict calls)
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
    # Convert to class label as required by sample_submission ("real"/"fake")
    pred_label = "fake" if fake_prob >= 0.5 else "real"
    rows.append([os.path.splitext(f)[0], pred_label])

sub = pd.DataFrame(rows, columns=["filename", "label"])
sub.to_csv("submission_final.csv", index=False)
print("Saved: submission_final.csv")
print(sub.head())