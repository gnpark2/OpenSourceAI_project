import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import webbrowser
import time

# 훈련된 모델 불러오기
model = tf.keras.models.load_model('spine_model_tf.keras')

# 폴더에 있는 모든 이미지 파일 경로 가져오기
image_folder = 'exampleOf_classify_spine'  # 실제 이미지가 있는 폴더 경로로 변경
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('jpg', 'jpeg', 'png'))]

#척추측만증 구분 0, 1
global divide

# 이미지를 분류하는 함수
def classify_spine_batch(image_paths):
    results = []
    global divide
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
        else:
            result = "이 사람은 척추측만증이 없습니다."

        results.append((image_path, result))

    return results

# 이미지들에 대한 결과 얻기
results = classify_spine_batch(image_files)

# 결과 출력
for image_path, result in results:
    print(f"{image_path}: {result}")

if divide == 1:
    print("척추측만증은 심하지 않다면 인체에 영향을 주지 않지만 심해지면 근육의 불균형과 관절에 문제가 생길 수 있습니다. 더 자세한 내용은 5초 후에 나오는 웹페이지에 있습니다.")
    time.sleep(5)
    webbrowser.open("https://www.amc.seoul.kr/asan/mobile/healthinfo/disease/diseaseDetail.do?contentId=32565")