# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math 
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df=web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end= '2019-12-17')
df


# %%
df.shape


# %%
plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price USD')


# %%
df = df['Close'].values
df = df.reshape(-1,1)

#The reshape allows you to add dimensions or change the number of elements in each dimension. We are using reshape(-1, 1) 
# because we have just one dimension in our array, so numpy will create the same number of our rows and 
# add one more axis: 1 to be the second dimension.


# %%
#split data
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])


# %%
#scale data
scaler = MinMaxScaler(feature_range= (0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.fit_transform(dataset_test)
dataset_train


# %%
# function to create datasets

def create_dataset(df):
    x= []
    y = []
    for i in range (50, df.shape[0]):
        x.append(df[i-50:i,0])
        y.append(df[i,0])
    x = np.array(x)
    y = np.array(y)

    return x, y


# %%
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)


# %%
#3d array reshapingfor LSTM layers
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))


# %%
model = Sequential()
model.add(LSTM(units=96,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units= 96, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.summary()


# %%
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# %%
model.compile(loss='mean_squared_error', optimizer='adam')


# %%
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('stock_prediction.h2')


# %%
model = load_model('stock_prediction.h2')


# %%
scaler.scale_


# %%
#scale_factor = 1/0.02893937
#y_train = y_train * scale_factor
#y_test = y_test * scale_factor


# %%
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1,1))

fig, ax = plt.subplots(figsize=(16,18))
#ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Original price')
plt.plot(predictions, color='cyan', label='Predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()


