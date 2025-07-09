# Deep Learning Education Repository

이 저장소는 딥러닝을 학습하기 위한 다양한 예제와 실습 코드들을 포함하고 있습니다. Perceptron부터 CNN, RNN, LSTM, GRU, Attention, Transformer까지 딥러닝 핵심 주제를 단계적으로 다룹니다.

---

## 📁 01_ANN

### 01_Perceptron
- `Perceptron-logicGate-example01.py`  
  단일 퍼셉트론을 활용한 논리 게이트 구현 예제입니다. (AND, OR 등)

### 02_Gradient_Descent
- `Learning-rate-example01.py` ~ `07.py`  
  학습률(learning rate)에 따른 그래디언트 하강법의 학습 차이를 실험하는 코드입니다.

### 03_ANN_Mnist
- `ANN-Mnist-example.py`  
  MNIST 데이터를 기반으로 한 단순한 ANN 모델 정의
- `ANN-Mnist-Run-example.py`  
  학습 실행 및 결과 출력용 스크립트
- `mnist.h5`  
  학습된 모델 저장 파일

---

## 📁 02_CNN

### 01_CNN_Example
- `CNN-example.py`, `CNN-example2.py`  
  기본 CNN 구조 예제
- `CNN-Run-example.py`, `CNN-Run-example2.py`  
  CNN 학습 및 평가 실행 코드
- `minst.h5`, `transfer_learing_flower.keras`, `history_flower`  
  학습된 모델 및 히스토리 파일

### 02_Medical_Images
- `Medical-images-example.py`  
  폐렴 이미지 분류를 위한 CNN 모델 정의
- `Medical-images-Run-example.py`  
  위 모델의 학습 실행 스크립트
- `pneumonia_model.h5`  
  학습된 폐렴 분류 모델
- `train_accuracy.png`, `train_loss.png`  
  훈련 정확도 및 손실 시각화 그래프

### 03_Computer_Vision
- `Convolution-lena-example.py`, `Edge-example.py`  
  이미지 처리 기초(CNN 필터, 엣지 감지 등)
- `lena.jpg`  
  실습용 테스트 이미지

---

## 📁 03_RNN_LSTM_GRU
- `RNN_LSTM_GRU.py`  
  순환 신경망(RNN), LSTM, GRU 모델의 구현 및 비교 예제

---

## 📁 04_S2S_Attention
- 향후 Sequence-to-Sequence 및 Attention 구조 예제 추가 예정

---

## 📁 05_Transformer
- 향후 Transformer 기반 모델 예제 추가 예정

---

## 📄 README.md
- 이 파일입니다. 전체 프로젝트 구조와 예제별 개요를 안내합니다.

---

## 📌 참고 사항
- 각 예제는 Python 기반이며, TensorFlow 또는 Keras 라이브러리를 사용합니다.
- `.h5`, `.keras` 파일은 학습된 모델의 저장본입니다.
- 실습 환경: Python 3.8 이상, TensorFlow 2.x 이상 권장
