#Based off of TensorFlow. Using a time series approach performed using a Recurring Neural Network.
#RNN is a neural network where the output of the previous step is fed as input to the current step. This is unlike traditional neural networks, where inputs and outputs are independent of each other.

from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

#loading dataset into pandas dataframe
tesla = pd.read_csv('TSLA.csv')
#print(tesla.head())
#print(tesla.shape)
tesla['Date'] = pd.to_datetime(tesla['Date'])
#tesla.info()
#tesla.describe()

#open close trends in Tesla stock
plt.plot(tesla['Date'],
         tesla['Open'],
         color="green",
         label="open")
plt.plot(tesla['Date'],
         tesla['Close'],
         color="red",
         label="close")
plt.title("Tesla Open-Close Stock")
plt.legend()
plt.show()

#trends in volume of Tesla Stock
plt.plot(tesla['Date'], tesla['Volume'])
plt.show()

#heatmap that analyzes the coorlation
sns.heatmap(tesla.corr(),annot = True, cbar = False)
plt.show()

#close prices of Tesla Stock from 2010-2020
tesla['Date'] = pd.to_datetime(tesla['Date'])
prediction = tesla.loc[(tesla['Date'] > datetime (2010, 1, 1)) & (tesla['Date'] < datetime(2020, 1, 1))]

plt.figure(figsize = (10,10))
plt.plot(tesla['Date'], tesla['Close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Tesla Stock Prices")
plt.show()

#preparing training set samples
tsla_close = tesla.filter(['Close'])
dataset = tsla_close.values
training = int(np.ceil(len(dataset) * .95))
print(training)
#START FROM DOWN HERE
#scaling the data
ss = StandardScaler()
ss = ss.fit_transform(dataset)

train_data = ss[0:int(training), :]

x_train = []
y_train = []

#considering 60 as batch size, create x_train and y_train
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train),\
                   np.array(y_train)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#building the model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(keras.layers.LSTM(units = 64))
model.add(keras.layers.Dense(128))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

#compiling the model
#the training process of the LSTM model
#0.0807 mean absolute error, close to perfect error score
from keras.metrics import RootMeanSquaredError
model.compile(optimizer = 'adam', loss = 'mae', metrics = RootMeanSquaredError())
history = model.fit(X_train, y_train, epochs = 20)

 #model evaluation -> evaluate its performance on validation data using different metrics
testing = ss[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(testing)):
    x_test.append(testing[i-60:i,0]) 

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred = model.predict(x_test)
  

#plotting predictions
train = tesla[:training]
test = tesla[:training]
test['Predictions'] = pred

plt.figure(figsize = (10, 8))
plt.plot(train['Close'], c="b")
plt.plot(test[['Close', 'Predictions']])
plt.title('Tesla Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()