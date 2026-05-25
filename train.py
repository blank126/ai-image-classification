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
EPOCHS = 20 # 전체 학습 반복 횟수
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
# 클래스 이름 출력
print("Class Names:", class_names)
# 클래스 개수 출력
print("Num Classes:", num_classes)


# 이미지 파일 경로를 저장할 리스트
image_paths = []
# 이미지 라벨을 저장할 리스트
labels = []


for label, class_name in enumerate(class_names):
    class_dir = os.path.join(BASE_DIR, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(file_path)
            labels.append(label)


# 이미지 경로 리스트를 numpy 배열로 변환
image_paths = np.array(image_paths)
# 라벨 리스트를 numpy 배열로 변환
labels = np.array(labels)


# Train 
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,                         
    labels,                              
    test_size=(VAL_RATIO + TEST_RATIO),  
    random_state=SEED,
    stratify=labels
)

# Validation + Test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED,
    stratify=temp_labels
)


# 학습 데이터 개수 출력
print("Train:", len(train_paths))
# 검증 데이터 개수 출력
print("Validation:", len(val_paths))
# 테스트 데이터 개수 출력
print("Test:", len(test_paths))


# ==========================================================
# 4. 데이터 전처리 (정규화 단계)
# ==========================================================
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    
    # 0~255 범위를 0.0~1.0 범위의 실수 값으로 변환 (정규화)
    image = tf.cast(image, tf.float32) / 255.0 
    
    return image, label


# 이미지 경로와 라벨 배열을 TensorFlow Dataset으로 변환
def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )
    ds = ds.map(load_image)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# 학습 데이터셋 생성
train_ds = make_dataset(train_paths, train_labels, shuffle=True)
# 검증 데이터셋 생성
val_ds = make_dataset(val_paths, val_labels)
# 테스트 데이터셋 생성
test_ds = make_dataset(test_paths, test_labels)


# 샘플 이미지를 출력 (필요없으면 주석 처리)
plt.figure(figsize=(10, 10))
for images, labels_batch in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")
plt.show()


# ==========================================================
# 5. 모델 설계 (★최종 업그레이드: 고도화된 CNN 구조 적용★)
# ==========================================================
# 이미지의 2D 공간 구조를 유지하며 특징을 추출하는 합성곱 신경망을 설계합니다.
model = tf.keras.Sequential(name='Advanced_CNN_Model')

# 입력층
model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

# [CNN 블록 1] 이미지의 거친 윤곽선 및 색상 패턴 추출
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization()) # 내부 공변량 변화를 방지하여 빠른 학습 가속화
model.add(layers.MaxPooling2D((2, 2))) # 해상도를 줄여 주요 정보만 압축
model.add(layers.Dropout(0.25))

# [CNN 블록 2] 눈, 코, 입 등 구체적인 동물의 형태적 특징 추출
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# [CNN 블록 3] 딥러닝 고차원 세부 특징 추출
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))


# [분류층 - DNN 연계] 2D 특징 맵이 완성되었으므로 최종 분류를 위해 일렬로 펼침
model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5)) # 전결합층의 강력한 과적합 방지 규제

# 최종 출력층 (7개 클래스 동물 분류)
model.add(layers.Dense(units=num_classes, activation='softmax', name='Output'))


# 최적화 및 하이퍼파라미터 컴파일 (안정적인 CNN 수렴을 위한 Adam 튜닝)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 변경된 CNN 모델 레이어 구조 출력
model.summary()


# 모델 학습
history = model.fit(
    train_ds,               # 학습 데이터
    validation_data=val_ds, # 검증 데이터
    epochs=EPOCHS,          # 학습 반복 횟수
    verbose=2               # epoch별 결과만 출력
)


# 검증 데이터로 성능 평가
val_loss, val_acc = model.evaluate(
    val_ds,
    verbose=0
)

# 검증 정확도 출력
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")


# 테스트 데이터로 최종 성능 평가
test_loss, test_acc = model.evaluate(
    test_ds,
    verbose=0
)

# 테스트 정확도 출력
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


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
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
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
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()