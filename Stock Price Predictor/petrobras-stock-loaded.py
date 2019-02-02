import numpy as np
from keras.models import model_from_json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file = open('stock-price.json', 'r')
networkStructure = file.read()
file.close()

regressor = model_from_json(networkStructure)
regressor.load_weights('stock-price-weights.h5')



dataset = pd.read_csv('petr4-treinamento.csv')
dataset = dataset.dropna()

normalizer = MinMaxScaler(feature_range = (0, 1))
trainingDataset = dataset.iloc[:, 1:7].values

previsionNormalizer = MinMaxScaler(feature_range = (0, 1))
previsionNormalizer.fit_transform(trainingDataset[:, 0:1])



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
