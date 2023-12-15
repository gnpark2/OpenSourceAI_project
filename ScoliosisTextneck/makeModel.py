import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# 훈련 데이터와 검증 데이터를 나누기 위한 함수
def split_train_validation_data(data_dir, validation_size=0.2, random_state=42):
    classes = os.listdir(data_dir)
    train_data = []
    validation_data = []

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        train_imgs, validation_imgs = train_test_split(images, test_size=validation_size, random_state=random_state)
        if class_name == 'normalScoliosis':
            train_data.extend([(img, 0) for img in train_imgs])  # 클래스 0
            validation_data.extend([(img, 0) for img in validation_imgs])  # 클래스 0
        elif class_name == 'Scoliosis':
            train_data.extend([(img, 1) for img in train_imgs])  # 클래스 1
            validation_data.extend([(img, 1) for img in validation_imgs])  # 클래스 1

    return train_data, validation_data

# 데이터셋 디렉토리 설정
data_dir = 'trainDataScoliosis'

# 훈련 데이터와 검증 데이터 나누기
train_data, validation_data = split_train_validation_data(data_dir, validation_size=0.2)

# 이미지 데이터 증강을 위한 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # 이미지 회전 각도 설정
    width_shift_range=0.2,  # 가로 방향 이동 범위 설정
    height_shift_range=0.2,  # 세로 방향 이동 범위 설정
    fill_mode='nearest'  # 이미지를 회전 또는 이동할 때 채울 픽셀 값 설정
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 로딩 및 증강
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    classes=['normalScoliosis', 'Scoliosis'],
    class_mode='binary'
)

# 간단한 CNN 모델 정의
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
num_epochs = 50
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs
)

# 학습된 모델 저장
model.save('spine_model_tf.keras')