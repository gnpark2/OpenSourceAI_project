import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import webbrowser
import time
import cv2

# 훈련된 모델 불러오기
model = tf.keras.models.load_model('spine_model_tf.keras')

# 폴더에 있는 모든 이미지 파일 경로 가져오기
image_folder = 'exampleOf_classify_spine'  # 실제 이미지가 있는 폴더 경로로 변경
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('jpg', 'jpeg', 'png'))]

#척추측만증 구분 0, 1
global divide, scoliosis_path

# 이미지를 분류하는 함수
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

def find_spinal_curve_length(image_path, user_height_cm):
    # 이미지 불러오기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 엣지 검출
    edges = cv2.Canny(image, 50, 150)
    #cv2.imwrite('a.jpg', edges)

    # 컨투어 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #컨투어의 bounding rectangle을 구하고 이를 이용하여 중앙과 중앙에 있는 컨투어의 길이 찾기
    spine_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(spine_contour)
    center_length = cv2.arcLength(spine_contour, closed=False)

    # 곡선 길이 초기화
    curve_length_pixel = 0.0

    # 모든 컨투어에 대해 곡선 길이 계산
    for contour in contours:
        curve_length_pixel += cv2.arcLength(contour, closed=False)

    # 사용자로부터 입력받은 상체의 실제 길이와 이미지에서 측정한 픽셀 길이를 비교하여 비율 계산
    pixel_to_cm_ratio = user_height_cm / h

    # 픽셀을 cm로 변환
    curve_length_cm = curve_length_pixel * pixel_to_cm_ratio

    return curve_length_cm, spine_contour, (x, y), center_length

# 이미지들에 대한 결과 얻기
results = classify_spine_batch(image_files)

# 결과 출력
for image_path, result in results:
    print(f"{image_path}: {result}")

# 척추측만증이 있다면 결과 추가 출력
if divide == 1:
    # 사용자로부터 실제 상체 길이 입력 받기
    user_height_cm = float(input("상체의 실제 길이를 입력하세요 (cm): "))

    # 곡선 길이 계산
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