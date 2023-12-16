import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# 훈련된 모델 불러오기
model = tf.keras.models.load_model('spine_model_tf.keras')

# 이미지를 분류하는 함수
def classify_spine(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] >= 0.5:
        result = "이 사람은 척추측만증이 있습니다."
    else:
        result = "이 사람은 척추측만증이 없습니다."

    return result

# 사용 예시
image_path = 'exampleOf_classify_spine/ex_scoliosis_False.jpg'  # 실제 이미지 경로로 바꿔주세요
result = classify_spine(image_path)
print(result)