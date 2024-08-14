

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from numpy import mean,std,array
from pandas import read_csv
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder
from keras.layers import LSTM, Dense, Dropout, TimeDistributed,Flatten
from keras.layers.convolutional import Conv1D,MaxPooling1D

"""Mount Google Drive"""

from google.colab import drive
drive.mount('/content/drive')

"""Download dataset"""

!wget -P /content/drive/MyDrive/FinalProjeect https://archive.ics.uci.edu/ml/machine-learning-databases/00506/casas-dataset.zip

"""Unzip necessary files"""

import os
import shutil
from tqdm import tqdm

zip_address = '/content/drive/MyDrive/FinalProjeect/casas-dataset.zip'
extracted_address = '/content/drive/MyDrive/FinalProjeect/casas-dataset-extracted'
unzip_command = '7z e {0} -o{1} {2} -x!{2}/{2}.rawdata.*'
rm_command = 'rm -r {0}'

try:
    os.mkdir(extracted_address)
except:
    shutil.rmtree(extracted_address)
    os.mkdir(extracted_address)

os.system(f'7z e {zip_address} -o{extracted_address} README.txt')
for i in tqdm(range(101, 131)):
    os.system(unzip_command.format(zip_address, f'{extracted_address}/csh{i}', f'csh{i}'))
    os.rmdir(f'{extracted_address}/csh{i}/csh{i}')

"""Read dataset"""

df = pd.read_csv('/content/drive/MyDrive/FinalProjeect/casas-dataset-extracted/csh101/csh101.ann.features.csv')
df

"""Find X ,y then :

*   preprocess data and one hot encoding for labels
*   use min max scaler to put inputs in range of 0,1
*   split dataset into train and test
"""

def getData():
  X = df.loc[:, df.columns != 'activity']
  y = df[['activity']]
  outputValues = array(y.values.ravel())
  lableEncoder = LabelEncoder()
  MappedValues = lableEncoder.fit_transform(outputValues)
  onehotEncoder = OneHotEncoder(sparse=False)
  MappedValues = MappedValues.reshape(len(MappedValues), 1)
  y = onehotEncoder.fit_transform(MappedValues)
  minMaxScaler = MinMaxScaler(feature_range = (0, 1))
  transforemdX = minMaxScaler.fit_transform(X)
  transformedY = np.array(y)
  train_X, test_X, train_y, test_y = train_test_split(transforemdX, transformedY, test_size=0.25, random_state=30)
  return train_X, test_X, train_y, test_y

"""# LSTM

Network with 1 layer LSTM
"""

train_X, test_X, train_y, test_y = getData()
lstmModel1 = Sequential()
lstmModel1.add(LSTM(units = 37, input_shape = (train_X.shape[1], 1)))
lstmModel1.add(Dropout(0.2))
lstmModel1.add(Dense(units = 1))
lstmModel1.compile(optimizer = 'adam', loss = 'mean_squared_error',  metrics=['accuracy'])
lstmModel1.fit(train_X, train_y, epochs = 10, batch_size = 30)

"""Model Accuracy"""

accuracy = lstmModel1.evaluate(test_X, test_y, batch_size=30)
print("Loss: , Accuracy:",accuracy)

"""Network with 2 layer LSTM"""

train_X, test_X, train_y, test_y = getData()
lstmModel2 = Sequential()
lstmModel2.add(LSTM(units = 37, return_sequences = True, input_shape = (train_X.shape[1], 1)))
lstmModel2.add(Dropout(0.2))
lstmModel2.add(LSTM(units = 37, return_sequences = True))
lstmModel2.add(Dropout(0.2))
lstmModel2.add(LSTM(units = 37))
lstmModel2.add(Dropout(0.2))
lstmModel2.add(Dense(units = 1))
lstmModel2.compile(optimizer = 'adam', loss = 'mean_squared_error',  metrics=['accuracy'])
lstmModel2.fit(train_X, train_y, epochs = 10, batch_size = 30)

"""Model Accuracy"""

accuracy = lstmModel2.evaluate(test_X, test_y, batch_size=30)
print("Loss: , Accuracy: ",accuracy)

"""# CNN-LSTM

CNN-LSTM Model with 1 layer CNN
"""

train_X, test_X, train_y, test_y = getData()
n_features,n_steps, n_length = 1, 4, 36

train_X = train_X.reshape((train_X.shape[0], 1, n_length, n_features))
test_X = test_X.reshape((test_X.shape[0], 1, n_length, n_features))

cnnLstmModel1 = Sequential()
cnnLstmModel1.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
cnnLstmModel1.add(TimeDistributed(Dropout(0.5)))
cnnLstmModel1.add(TimeDistributed(MaxPooling1D(pool_size=2)))
cnnLstmModel1.add(TimeDistributed(Flatten()))
cnnLstmModel1.add(LSTM(units = 36, return_sequences = True, input_shape = (train_X.shape[1], 1)))
cnnLstmModel1.add(Dropout(0.2))
cnnLstmModel1.add(LSTM(units = 36, return_sequences = True))
cnnLstmModel1.add(Dropout(0.2))
cnnLstmModel1.add(LSTM(units = 36))
cnnLstmModel1.add(Dropout(0.2))
cnnLstmModel1.add(Dense(units= 1 ,activation="softmax"))
cnnLstmModel1.compile(optimizer = 'adam', loss = 'mean_squared_error',  metrics=['accuracy'])
cnnLstmModel1.fit(train_X, train_y, epochs = 10, batch_size = 30)

"""Model Accuracy"""

accuracy = cnnLstmModel1.evaluate(test_X, test_y, batch_size=30)
print("Loss: , Accuracy: ",accuracy)

"""CNN-LSTM Model with 2 layer CNN"""

train_X, test_X, train_y, test_y = getData()
n_features,n_steps, n_length =1, 4, 36

train_X = train_X.reshape((train_X.shape[0], 1, n_length, n_features))
test_X = test_X.reshape((test_X.shape[0], 1, n_length, n_features))

cnnLstmModel2 = Sequential()
cnnLstmModel2.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
cnnLstmModel2.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
cnnLstmModel2.add(TimeDistributed(Dropout(0.5)))
cnnLstmModel2.add(TimeDistributed(MaxPooling1D(pool_size=2)))
cnnLstmModel2.add(TimeDistributed(Flatten()))
cnnLstmModel2.add(LSTM(units = 36, return_sequences = True, input_shape = (train_X.shape[1], 1)))
cnnLstmModel2.add(Dropout(0.2))
cnnLstmModel2.add(LSTM(units = 36, return_sequences = True))
cnnLstmModel2.add(Dropout(0.2))
cnnLstmModel2.add(LSTM(units = 36))
cnnLstmModel2.add(Dropout(0.2))
cnnLstmModel2.add(Dense(units= 35 ,activation="softmax"))
cnnLstmModel2.compile(optimizer = 'adam', loss = 'mean_squared_error',  metrics=['accuracy'])
fittedModelCnn=cnnLstmModel2.fit(train_X, train_y, epochs = 30, batch_size = 30)

"""Model Accuracy"""

accuracy = cnnLstmModel2.evaluate(test_X, test_y, batch_size=30)
print("Loss: , Accuracy: ",accuracy)

"""# Enhance Results

Find important features
"""

importance = fittedModelCnn.feature_importances_
feat_importances = pd.Series(importance, index=X.columns)
feat_importances=feat_importances.sort_values(ascending=True)
pyplot.figure(figsize=(20,20))
feat_importances.plot(kind='barh')
for index, score in feat_importances.items():
    print(f"Feature : {index}, Score : {score}")