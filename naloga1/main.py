import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Read data
data = pd.read_fwf("auto-mpg.data", names=["mpg", "c", "2", "3", "4", "5", "6", "7", "n"], header=None, na_values=["?"]) #mpg -> class attribute
data.drop(["c", "n"], axis=1, inplace=True)
data.dropna(inplace=True)
#encode categorical attribute
data_encoded = pd.get_dummies(data, columns=["7"], prefix=["categorical"], drop_first=True)
#data_encoded.drop(["7"])
#print(data_encoded)

dataNp = data_encoded.to_numpy()
#Shuffle data rows
np.random.shuffle(dataNp)
classNp = np.copy(dataNp[:, 0])

dataNp[:, 0] = 1
#print(dataNp[:3, :])

rows = dataNp.shape[0]
"""
dataNp = np.c_[dataNp, np.zeros(rows)]

for i in range(rows):
    if(dataNp[i, -2] == 1):
        dataNp[i, -2] = 0
    elif(dataNp[i, -2] == 2):
        dataNp[i, -2] = 1
    elif(dataNp[i, -2] == 3):
        dataNp[i, -2] = 0
        dataNp[i, -1] = 1
"""


#train test split
splitRow = 270
trainData = np.copy(dataNp[splitRow:, :])
trainClass = np.copy(classNp[splitRow:])

testData = np.copy(dataNp[:splitRow, :])
testClass = np.copy(classNp[:splitRow])


#regression
prediction = np.linalg.inv(trainData.T @ trainData) @ trainData.T @ trainClass
#print(prediction.shape)
#print(prediction)

#Scikit learn
reg = LinearRegression().fit(trainData, trainClass)
#print(reg.coef_)
predSklearn = reg.predict(testData)


#Testing
predictions = testData @ prediction

residualError = testClass - predictions

residualsSklearn = testClass - predSklearn

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(residualError)), residualError, 'o')
plt.title("My implementation")
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(residualsSklearn)), residualsSklearn, 'o')
plt.title("Sklearn implementation")
plt.show()

print(f"MSE: {sum(np.square(residualError))/len(residualError)}")

#Residuals do not appear to be distributed at random -> take log of class attribute

#logarithmic transformation

classNp = np.log(classNp)

#shuffle data rows
splitRow = 270
trainData = np.copy(dataNp[splitRow:, :])
trainClass = np.copy(classNp[splitRow:])

testData = np.copy(dataNp[:splitRow, :])
testClass = np.copy(classNp[:splitRow])

#regression
prediction = np.linalg.inv(trainData.T @ trainData) @ trainData.T @ trainClass
#print(prediction.shape)
#print(prediction)

#Scikit learn
reg = LinearRegression().fit(trainData, trainClass)
#print(reg.coef_)
predSklearn = reg.predict(testData)


#Testing
predictions = testData @ prediction

residualError = testClass - predictions

residualsSklearn = testClass - predSklearn

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(residualError)), residualError, 'o')
plt.title("My implementation")
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(residualsSklearn)), residualsSklearn, 'o')
plt.title("Sklearn implementation")
plt.show()

print(f"MSE: {sum(np.square(residualError))/len(residualError)}")