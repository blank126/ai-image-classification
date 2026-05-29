import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Config

BASE_DIR = "./dataset"  # 데이터셋 폴더 경로

IMG_HEIGHT = 96 # 입력 이미지 높이
IMG_WIDTH = 96  # 입력 이미지 너비
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

SEED = 123  # 랜덤 시드값

BATCH_SIZE = 32 # 한 번에 학습할 이미지 개수
EPOCHS = 50 # 조기종료를 믿고 에포크를 50으로 충분히 늘립니다.
TRAIN_RATIO = 0.7   # 학습 데이터 비율
VAL_RATIO = 0.15    # 검증 데이터 비율
TEST_RATIO = 0.15   # 테스트 데이터 비율


# 폴더 이름을 클래스 이름으로 사용
class_names = sorted([
    name for name in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, name))
])

# 클래스 개수 저장
num_classes = len(class_names)
print("Class Names:", class_names)
print("Num Classes:", num_classes)


# 이미지 파일 경로 및 라벨 저장
image_paths = []
labels = []

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(BASE_DIR, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(file_path)
            labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)


# Train / Validation / Test Split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,                         
    labels,                              
    test_size=(VAL_RATIO + TEST_RATIO),  
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED,
    stratify=temp_labels
)

print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))


# 데이터 전처리 (정규화)
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label


def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_image)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)


# ==========================================================
# [수정 구간 1] 데이터 증강 변형 강도 최적화 (0.15 -> 0.1)
# ==========================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),   # 좌우 반전
    layers.RandomRotation(0.1),        # 최대 10도 회전으로 형태 보존
    layers.RandomZoom(0.1),            # 최대 10% 확대/축소로 왜곡 방지
], name="Data_Augmentation")


# ==========================================================
# 5. 모델 설계 (CNN 구조 + 데이터 증강 연계)
# ==========================================================
model = tf.keras.Sequential(name='Advanced_Anti_Overfitting_CNN')

# 입력층 및 데이터 증강층 적용
model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(data_augmentation) 

# CNN 블록 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# CNN 블록 2
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# CNN 블록 3
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3)) 

# 분류층
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

# 출력층
model.add(layers.Dense(units=num_classes, activation='softmax', name='Output'))


# 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ==========================================================
# [수정 구간 2] 조기 종료 대기 시간 연장 및 가변 학습률 추가
# ==========================================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',        
    patience=12,               # 5에서 12로 변경 (모델이 충분히 탐색하도록 기회 부여)
    restore_best_weights=True  # 가장 성적이 좋았던 순간의 가중치 복원
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                # 정체기 진입 시 학습률을 절반으로 낮춰 정밀 조절
    patience=4,                # 4번의 에포크 동안 개선이 없으면 발동
    verbose=1
)


# 모델 학습
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr], # 두 콜백 함수 연계 실행
    verbose=2
)


# 검증 데이터 평가
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\nBest Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Best Validation Loss: {val_loss:.4f}")

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")


# 실제 라벨 및 예측 결과 저장
y_true = []
y_pred = []

for images, labels_batch in test_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_true.extend(labels_batch.numpy())
    y_pred.extend(preds)

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 학습 곡선 시각화
epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
save_path_loss = "./loss_graph.png"
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()