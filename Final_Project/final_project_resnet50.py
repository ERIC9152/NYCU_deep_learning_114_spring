import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# 1. GPU 設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 已就緒")
    except RuntimeError as e: print(e)

# 2. 參數與路徑
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
train_dir = "/mnt/d/DL/Final_Project/train"
test_dir = "/mnt/d/DL/Final_Project/test"

# 3. 資料增強
train_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.1).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# 4. ResNet50 模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = True if "conv5" in layer.name else False

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
model = Model(inputs=base_model.input, outputs=Dense(1, activation='sigmoid')(x))
model.compile(optimizer=Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 5. Checkpoint
checkpoint_path = "/mnt/d/DL/Final_Project/resnet50_best_model.h5"
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='loss', patience=6, restore_best_weights=True, verbose=1)
]

print("\n開始進行 ResNet50 訓練...")
model.fit(train_generator, epochs=25, callbacks=callbacks)

# 6. 測試集預測 (符合 sample_submission 格式)
print("\n正在進行預測...")

# 獲取測試檔案並按數字大小排序 (確保 1, 2, 3... 順序)
files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
files.sort(key=lambda x: int(os.path.splitext(x)[0]))

results = []
for i, f in enumerate(files):
    img_path = os.path.join(test_dir, f)
    img = img_to_array(load_img(img_path, target_size=IMG_SIZE)) / 255.0
    p = model.predict(np.expand_dims(img, axis=0), verbose=0)
    
    # 類別判斷
    label = 'real' if p[0][0] > 0.5 else 'fake'
    # 提取純數字檔名
    file_id = os.path.splitext(f)[0]
    
    results.append({"filename": file_id, "label": label})
    if (i+1) % 1000 == 0: print(f"已完成 {i+1} / {len(files)} 張")

# 存檔
submission = pd.DataFrame(results)
output_path = '/mnt/d/DL/Final_Project/resnet50_predictions.csv'
submission.to_csv(output_path, index=False)
print(f"預測完成！結果已存至: {output_path}")