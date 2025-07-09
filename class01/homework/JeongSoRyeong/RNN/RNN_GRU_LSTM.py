import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

#범위를 0~1로 normalized
def MinMaxScaler(data):
    #최소값과 최댓값을 이용하여 0~1 값으로 변환
    numberator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    #0으로 나누기 에러가 발생하지 않도록 매우 작은 값을 더해서 나눔
    return numberator / (denominator + 1e-7)


df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', 'High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

#두 데이터를 리스트 형태로 저장
x = dfx.values.tolist() #open, high, log, volume 데이터
y = dfy.values.tolist() #close 데이터

window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] #다음 날 종가는 포함되지 않음
    _y = y[i + window_size]     #다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

#훈련 데이터 크기 70%, 검증 데이터 20%, 테스트
train_size = int(len(data_y)*0.7)
val_size = int(len(data_y)*0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])

test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])

print('훈련 데이터의 크기 : ', train_x.shape, train_y.shape)
print('검증 데이터의 크기 : ', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 : ', test_x.shape, test_y.shape)

# 데이터셋 크기
sizes = [len(train_x), len(val_x), len(test_x)]
labels = ['Train', 'Validation', 'Test']


#RNN 모델
model_RNN = Sequential()
model_RNN.add(SimpleRNN(units=20, activation='tanh',
                    return_sequences=True,
                    input_shape=(10,4)))
model_RNN.add(Dropout(0.1))
model_RNN.add(SimpleRNN(units=20, activation='tanh'))
model_RNN.add(Dropout(0.1))
model_RNN.add(Dense(units=1))
# model_RNN.summary()

model_RNN.compile(optimizer='adam',
             loss ='mean_squared_error')
history = model_RNN.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=70, batch_size=30)

#GRU
model_GRU = Sequential()
model_GRU.add(GRU(units=20, activation='tanh',
                    return_sequences=True,
                    input_shape=(10,4)))
model_GRU.add(Dropout(0.1))
model_GRU.add(GRU(units=20, activation='tanh'))
model_GRU.add(Dropout(0.1))
model_GRU.add(Dense(units=1))
# model_GRU.summary()

model_GRU.compile(optimizer='adam',
             loss ='mean_squared_error')
history = model_GRU.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=70, batch_size=30)

#LSTM
model_LSTM = Sequential()
model_LSTM.add(LSTM(units=20, activation='tanh',
                    return_sequences=True,
                    input_shape=(10,4)))
model_LSTM.add(Dropout(0.1))
model_LSTM.add(LSTM(units=20, activation='tanh'))
model_LSTM.add(Dropout(0.1))
model_LSTM.add(Dense(units=1))
# model_LSTM.summary()

model_LSTM.compile(optimizer='adam',
             loss ='mean_squared_error')
history = model_LSTM.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=70, batch_size=30)

# 테스트 데이터에 대한 예측값 생성
pred_rnn = model_RNN.predict(test_x)
pred_gru = model_GRU.predict(test_x)
pred_lstm = model_LSTM.predict(test_x)

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(test_y)), test_y, label='Actual', color='black', linewidth=2)
plt.plot(np.arange(len(test_y)), pred_rnn, label='RNN', color='red', linestyle='--')
plt.plot(np.arange(len(test_y)), pred_gru, label='GRU', color='blue', linestyle='--')
plt.plot(np.arange(len(test_y)), pred_lstm, label='LSTM', color='green', linestyle='--')
plt.title('Stock Price Prediction: Actual vs RNN/GRU/LSTM')
plt.xlabel('Time')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()