# 오픈소스AI응용 기말고사 최종 프로젝트
이 프로젝트는 캐글(Kaggle)에 있는 사람의 등 사진 데이터를 통해 척추측만증 유무를 분류하는 모델을 만들고, 학습된 모델을 이용하여 새로운 등 사진을 받았을 때, 척추측만증의 유무 판단.
추가로 척추측만증이라고 판단되면 자세한 진단을 받아볼 것을 요구하는 코멘트와 척추측만증 관련 내용이 있는 서울아산병원 페이지를 팝업.
그리고 cv2(openCV)를 통해 척추를 인식하여 곡선의 길이 도출.
사용자에게 상반신의 길이를 입력받고 곡선의 길이를 직선으로 바꿨을 때의 비율을 이용하여 교정 시 키를 분석.

# python 및 library version
주요 요구사항 : python 3.11.3, keras : 2.14.0, tensorflow : 2.14.0, opencv(cv2) : 4.8.1.78
python 3.11.3
pip                          23.3.2
numpy                        1.26.0
keras                        2.14.0
Keras-Preprocessing          1.1.2
tensorboard                  2.14.1
tensorboard-data-server      0.7.2
tensorflow                   2.14.0
tensorflow-estimator         2.14.0
tensorflow-hub               0.15.0
tensorflow-intel             2.14.0
tensorflow-io-gcs-filesystem 0.31.0
opencv-contrib-python        4.8.1.78
opencv-python                4.8.1.78
contourpy                    1.1.1
absl-py                      2.0.0       
astunparse                   1.6.3       
attrs                        23.1.0      
cachetools                   5.3.2
certifi                      2023.7.22
cffi                         1.16.0
chardet                      5.2.0
charset-normalizer           3.3.2
cycler                       0.12.0
flatbuffers                  23.5.26
fonttools                    4.43.0
gast                         0.5.4
google-auth                  2.23.4
google-auth-oauthlib         1.0.0
google-pasta                 0.2.0
grpcio                       1.59.2
h5py                         3.10.0
idna                         3.4
joblib                       1.3.2
kiwisolver                   1.4.5
libclang                     16.0.6
Markdown                     3.5.1
MarkupSafe                   2.1.3
matplotlib                   3.8.0
mediapipe                    0.10.9
ml-dtypes                    0.2.0
oauthlib                     3.2.2
opt-einsum                   3.3.0
packaging                    23.2
pandas                       2.1.1
Pillow                       10.0.1
protobuf                     3.20.3
pyasn1                       0.5.0
pyasn1-modules               0.3.0
pycparser                    2.21
pyparsing                    3.1.1
python-dateutil              2.8.2
pytz                         2023.3.post1
requests                     2.31.0
requests-oauthlib            1.3.1
rsa                          4.9
scikit-learn                 1.3.2
scipy                        1.11.3
setuptools                   65.5.1
six                          1.16.0
sounddevice                  0.4.6
termcolor                    2.3.0
threadpoolctl                3.2.0
typing_extensions            4.8.0
tzdata                       2023.3
urllib3                      2.0.7
Werkzeug                     3.0.1
wheel                        0.38.4
wrapt                        1.14.1

# 코드설명
1. Import Libraries
```python
import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import webbrowser
import time
import cv2

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
```

2. Data Load
```python
data_dir = 'trainDataScoliosis'
```

3. data split(train and valid)
```python
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

train_data, validation_data = split_train_validation_data(data_dir, validation_size=0.2)
```

4. Data augmentation
```python
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

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    classes=['normalScoliosis', 'Scoliosis'],
    class_mode='binary'
)
```

5. Define CNN and compile model
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_epochs = 50
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs
)
```
```python
model.save('spine_model_tf.keras')
```

6. Load model and data
```python
model = tf.keras.models.load_model('spine_model_tf.keras')

image_folder = 'exampleOf_classify_spine'  # 실제 이미지가 있는 폴더 경로로 변경
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('jpg', 'jpeg', 'png'))]
```

7. classify image to scoliosis or not
```python
def classify_spine_batch(image_paths):
    results = []
    global divide, scoliosis_path

    for image_path in image_paths:
        # 이미지 크기를 모델이 예상하는 크기로 조절
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)

        if prediction[0][0] >= 0.5:
            result = "이 사람은 척추측만증이 있습니다."
            divide = 1
            scoliosis_path = image_path
        else:
            result = "이 사람은 척추측만증이 없습니다."

        results.append((image_path, result))

    return results
```

8. Calculate spinal curve length
```python
def find_spinal_curve_length(image_path, user_height_cm):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(image, 50, 150)
    #cv2.imwrite('a.jpg', edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spine_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(spine_contour)
    center_length = cv2.arcLength(spine_contour, closed=False)

    curve_length_pixel = 0.0

    for contour in contours:
        curve_length_pixel += cv2.arcLength(contour, closed=False)

    pixel_to_cm_ratio = user_height_cm / h

    curve_length_cm = curve_length_pixel * pixel_to_cm_ratio

    return curve_length_cm, spine_contour, (x, y), center_length
```

9. Result of data
```python
results = classify_spine_batch(image_files)

for image_path, result in results:
    print(f"{image_path}: {result}")

if divide == 1:
    user_height_cm = float(input("상체의 실제 길이를 입력하세요 (cm): "))

    length, spine, center, center_length = find_spinal_curve_length(scoliosis_path, user_height_cm)

    image_with_spine_contour = cv2.drawContours(cv2.imread(image_path), [spine], -1, (0, 255, 0), 2)
    cv2.circle(image_with_spine_contour, center, 5, (0, 0, 255), -1)
    print(f"Center Length of Contour: {center_length}")
    cv2.imshow('Image with Spine Contour and Center', image_with_spine_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Spinal curve length: {length}")

    print("척추측만증은 심하지 않다면 인체에 영향을 주지 않지만 심해지면 근육의 불균형과 관절에 문제가 생길 수 있습니다. 더 자세한 내용은 5초 후에 나오는 웹페이지에 있습니다.")
    time.sleep(5)
    webbrowser.open("https://www.amc.seoul.kr/asan/mobile/healthinfo/disease/diseaseDetail.do?contentId=32565")
```

# 실행방법
1. library를 설치한다. (필요시 버전을 맞춰서 설치한다.)
2. 측정할 상반신 등 사진(나체)을 exampleOf_classify_spine폴더에 넣는다. (사진은 jpg, jpeg, png 형식이어야 한다. 해당 폴더에 이미 있는 테스트 이미지 파일들은 삭제해도 무관하다.)
3. 코드를 실행시킨다.
