import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files

dataset_train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

print(dataset_train.shape)
#print(dataset_train.tail(10))
print(training_set)


# melakukan normalisasi 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)

# ========================================================================================================
# Now, we create a data structure with 60 timesteps and one output as an Array of x_train and y_train.
# Mempersiapakan data untuk masuk ke dalam arsitektur RNN 

X_train = []      # inisialisasi X data latih - dengan cara membuat list kosong 
y_train = []      # inisisalisasi Y data training - dengan cara membuat list kosong 

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)
print(y_train.shape)


#Here we have done reshaping of x_train data.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train)

#===================================== Merancang Arsitektur RNN ==============================

#import library untuk RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units = 120, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 70))
regressor.add(Dropout(0.3))


#layer output 
regressor.add(Dense(units = 1))
#====================== selesai perancangan arsitektur ===========================#

# proses training
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)


#membaca data testing
dataset_test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Melakukan visualisasi 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
