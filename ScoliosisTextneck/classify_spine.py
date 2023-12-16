import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# 훈련된 모델 불러오기
model = tf.keras.models.load_model('spine_model_tf.keras')

# 폴더에 있는 모든 이미지 파일 경로 가져오기
image_folder = 'exampleOf_classify_spine'  # 실제 이미지가 있는 폴더 경로로 변경
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
               file.endswith(('jpg', 'jpeg', 'png'))]


# 이미지를 분류하는 함수
def classify_spine_batch(image_paths):
    results = []
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)

        if prediction[0][0] >= 0.5:
            result = "이 사람은 척추측만증이 있습니다."
        else:
            result = "이 사람은 척추측만증이 없습니다."

        results.append((image_path, result))

    return results


# 이미지들에 대한 결과 얻기
results = classify_spine_batch(image_files)

# 결과 출력
for image_path, result in results:
    print(f"{image_path}: {result}")