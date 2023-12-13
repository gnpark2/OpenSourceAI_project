import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# 데이터셋 디렉토리 설정
train_dir = 'trainData'
validation_dir = 'validData'

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
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# 간단한 CNN 모델 정의
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
num_epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# 학습된 모델 저장
model.save('spine_model_tf.h5')