import FinanceDateReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,GRU, LSTM, Dropout

#DataSet Preparation
df = fdr.DataReader('005930','2018-05-04','2020-01-22')
dfx = df[['Open','High','Low','Volume','Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y)* 0.7)

val_size = int(len(data_y)* 0.2)

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

#GRU
model = Sequential()
model.add(GRU(units=20, activation='tanh',
                            return_sequences=True,
                            input_shape=(10,4)))
model.add(Dropout(0.1))
model.add(GRU(units=20,activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam',
                loss='mean_squared_error')
history = model.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=70, batch_size=30)
