import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 모델 로드
model = tf.keras.models.load_model('./mnist.h5')

# 데이터셋 로드
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

# 정규화
f_image_train = f_image_train / 255.0
f_image_test = f_image_test / 255.0

# 예측
num = 10
predict = model.predict(f_image_test[:num])
print("* Prediction:", np.argmax(predict, axis=1))
print("* Label:     ", f_label_test[:num])

# 예측 결과
pred_probs = predict          # softmax 확률값
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = f_label_test[:num]

plt.figure(figsize=(14, 5))

for i in range(num):
    plt.subplot(2, 5, i + 1)
    plt.imshow(f_image_test[i], cmap='gray')
    plt.xticks([]); plt.yticks([])
    
    pred = pred_labels[i]
    true = true_labels[i]
    prob = pred_probs[i][pred]  # 예측된 클래스의 확률

    # 맞았는지 색상으로 구분
    color = 'green' if pred == true else 'red'

    plt.xlabel(f"Pred: {pred} ({prob:.2f})\nTrue: {true}", color=color)

plt.tight_layout()
plt.show()
