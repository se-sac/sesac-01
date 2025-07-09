import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]



x=dfx.values.tolist()
y=dfy.values.tolist()

window_size = 10
data_x = []
data_y = []

for i in range(len(y) - window_size):
    _x = x[i : i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)


train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])

test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])

print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)


#RNN 모델
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=20, activation='tanh', 
                    return_sequences=True,
                    input_shape=(10,4))) # sequence length 10, input dimension 4
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(units=20, activation='tanh')) 
rnn_model.add(Dropout(0.1))
rnn_model.add(Dense(units=1))
rnn_model.summary()

rnn_model.compile(optimizer='adam', loss='mean_squared_error')
history_rnn = rnn_model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=70, batch_size=30)


# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


#GRU Model
gru_model = Sequential()
gru_model.add(GRU(units=20, activation='tanh', return_sequences=True, input_shape=(10,4)))
gru_model.add(Dropout(0.1))
gru_model.add(GRU(units=20, activation='tanh'))
gru_model.add(Dropout(0.1))
gru_model.add(Dense(units=1))
gru_model.summary()

gru_model.compile(optimizer='adam', loss='mean_squared_error')
history_gru = gru_model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=70, batch_size=30)



#LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=20, activation='tanh', return_sequences=True, input_shape=(10,4)))
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(units=20, activation='tanh'))
lstm_model.add(Dense(units=1))
lstm_model.summary()

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
history_lstm = lstm_model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=70, batch_size=30)

# RNN prediction
rnn_pred = rnn_model.predict(test_x)  # or model_rnn.predict() if using keras
# But since you're using Keras models, you should do:
#rnn_pred = rnn_model.predict(test_x)

# GRU prediction
gru_pred = gru_model.predict(test_x)

# LSTM prediction
lstm_pred = lstm_model.predict(test_x)

# 2. Convert predictions to 1D array
rnn_pred = rnn_pred.flatten()
gru_pred = gru_pred.flatten()
lstm_pred = lstm_pred.flatten()

# 3. Actual close prices (test_y)
actual = test_y.flatten()

# 4. Plot
plt.figure(figsize=(15,6))
# x-axis: time index
time_idx = range(len(actual))

plt.plot(time_idx, actual, label='Actual Close Price', color='black')
plt.plot(time_idx, rnn_pred, label='RNN Predicted', linestyle='--')
plt.plot(time_idx, gru_pred, label='GRU Predicted', linestyle='--')
plt.plot(time_idx, lstm_pred, label='LSTM Predicted', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Samsung Actual vs Predicted Stock Prices')
plt.legend()
plt.show()
