import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# ==========================================
# 1. GPU 最佳化設定
# ==========================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 已就緒")
    except RuntimeError as e: print(e)

# ==========================================
# 2. 參數設定
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
train_dir = "/mnt/d/DL/Final_Project/train"
test_dir = "/mnt/d/DL/Final_Project/test"

# ==========================================
# 3. 資料讀取
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ==========================================
# 4. 建立 VGG16 模型
# ==========================================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

for layer in base_model.layers:
    if "block5" in layer.name or "block4" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 5. 訓練回調
# ==========================================
checkpoint_path = "/mnt/d/DL/Final_Project/vgg16_best_model.h5"
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
]

print("\n開始進行 VGG16 訓練...")
model.fit(train_generator, epochs=30, callbacks=callbacks)

# ==========================================
# 6. 測試集預測 (符合 sample_submission 格式)
# ==========================================
print("\n正在逐一進行預測...")

# 獲取測試檔案並按數字大小排序
files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
files.sort(key=lambda x: int(os.path.splitext(x)[0]))

results = []
for i, filename in enumerate(files):
    img_path = os.path.join(test_dir, filename)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    label = 'real' if pred[0][0] > 0.5 else 'fake'
    file_id = os.path.splitext(filename)[0]
    
    results.append({"filename": file_id, "label": label})
    
    if (i + 1) % 1000 == 0:
        print(f"進度: {i+1} / {len(files)}")

# 儲存結果
submission = pd.DataFrame(results)
output_file = '/mnt/d/DL/Final_Project/vgg16_predictions.csv'
submission.to_csv(output_file, index=False)
print(f"\n[完成] 結果已存至: {output_file}")