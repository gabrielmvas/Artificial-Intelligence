from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


dataset = pd.read_csv('petr4-treinamento.csv')
dataset = dataset.dropna()

trainingDataset = dataset.iloc[:, 1:7].values

normalizer = MinMaxScaler(feature_range = (0, 1))
normalizedTrainingDataset = normalizer.fit_transform(trainingDataset)

previsionNormalizer = MinMaxScaler(feature_range = (0, 1))
previsionNormalizer.fit_transform(trainingDataset[:, 0:1])

predictors = []
realPrice = []
for i in range(90, 1242):
    predictors.append(normalizedTrainingDataset[i-90:i, 0:6])
    realPrice.append(normalizedTrainingDataset[i, 0])
predictors, realPrice = np.array(predictors), np.array(realPrice)

regressor = Sequential()
regressor.add(LSTM( units = 100,
                   return_sequences = True,
                   input_shape = (predictors.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM( units = 50, return_sequences = True ))
regressor.add(Dropout(0.3))

regressor.add(LSTM( units = 50, return_sequences = True ))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add( Dense(units = 1, activation = 'linear') )

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'weights.h5', monitor = 'loss', save_best_only = True, verbose = 1)

regressor.fit(predictors, realPrice, epochs = 200, batch_size = 32,
              callbacks = [es, rlr, mcp])


regressorJSON = regressor.to_json()
with open('stock-price.json', 'w') as json_file:
    json_file.write(regressorJSON)
regressor.save_weights('stock-price-weights.h5')

testDataset = pd.read_csv('petr4-teste.csv')
testRealPrice = testDataset.iloc[:, 1:2].values
frames = [dataset, testDataset]
completeDataset = pd.concat(frames)
completeDataset = completeDataset.drop('Date', axis = 1)

entries = completeDataset[len(completeDataset) - len(testDataset) - 90:].values
entries = normalizer.transform(entries)


xTest = []
for i in range(90, 112):
    xTest.append(entries[i-90:i, 0:6])
xTest = np.array(xTest)

previsions = regressor.predict(xTest)
previsions = previsionNormalizer.inverse_transform(previsions)

previsions.mean()
testRealPrice.mean()


plt.plot(testRealPrice, color = 'red', label = 'Real Price')
plt.plot(previsions, color = 'blue', label = 'Previsions')
plt.title('Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
