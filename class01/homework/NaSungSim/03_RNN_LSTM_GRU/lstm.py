import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# MinMaxScaler 함수 수정
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 데이터 불러오기
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')

# 필요한 컬럼 선택
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]

# 정규화
dfx = MinMaxScaler(dfx)

# X, Y 분리
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]

# 리스트로 변환
x = dfx.values.tolist()
y = dfy.values.tolist()

# 시퀀스 데이터 생성
window_size = 10
data_x = []
data_y = []

for i in range(len(y) - window_size):
    _x = x[i:i+window_size]
    _y = y[i+window_size]
    data_x.append(_x)
    data_y.append(_y)

# 넘파이 배열로 변환
data_x = np.array(data_x)
data_y = np.array(data_y)

# 데이터셋 분할 (70% train / 20% val / 10% test)
train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
test_size = len(data_y) - train_size - val_size

train_x = data_x[:train_size]
train_y = data_y[:train_size]

val_x = data_x[train_size:train_size+val_size]
val_y = data_y[train_size:train_size+val_size]

test_x = data_x[train_size+val_size:]
test_y = data_y[train_size+val_size:]

print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)


# 모델 생성
model = Sequential()

# 첫 번째 LSTM 레이어
model.add(LSTM(units=20, activation='tanh',
               return_sequences=True,     # 다음 LSTM으로 시퀀스 전체 전달
               input_shape=(10, 4)))      # 입력 데이터: (타임스텝 10, 특성 4)

# 드롭아웃 (과적합 방지)
model.add(Dropout(0.1))

# 두 번째 LSTM 레이어
model.add(LSTM(units=20, activation='tanh'))

# 드롭아웃
model.add(Dropout(0.1))

# 출력 레이어
model.add(Dense(units=1))  # 회귀값 1개 출력

# 모델 구조 출력
model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='mean_squared_error')

# 모델 학습
history = model.fit(train_x, train_y,
                    validation_data=(val_x, val_y),
                    epochs=70,
                    batch_size=30)
# 예측
train_pred = model.predict(train_x)
val_pred = model.predict(val_x)
test_pred = model.predict(test_x)


# 그래프 그리기
plt.figure(figsize=(14, 6))

# 훈련 데이터
plt.subplot(3, 1, 1)
plt.plot(train_y, label='Actual')
plt.plot(train_pred, label='Predicted')
plt.title('Train Data')
plt.legend()

# 검증 데이터
plt.subplot(3, 1, 2)
plt.plot(val_y, label='Actual')
plt.plot(val_pred, label='Predicted')
plt.title('Validation Data')
plt.legend()

# 테스트 데이터
plt.subplot(3, 1, 3)
plt.plot(test_y, label='Actual')
plt.plot(test_pred, label='Predicted')
plt.title('Test Data')
plt.legend()

plt.tight_layout()
plt.show()

# Loss 그래프
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

