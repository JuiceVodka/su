import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd

#load data
data = pd.read_csv("communities+and+crime/communities.data", na_values=["?"], header=None)
data.drop(list(range(5)), axis=1, inplace=True)
data.interpolate(axis=1, inplace=True)

data = data.to_numpy()

target = data[:, -1:]

data = data[:, :-1]

#train test split
train_data = data[:1500, :]
test_data = data[1500:, :]

train_class = target[:1500]
test_class = target[1500:]


#fit the model using scikit learn linear regression with L2 regularization
reg = Ridge(alpha=0.1)
reg.fit(train_data, train_class)

#predict
pred = reg.predict(test_data)
print(pred.shape)
#calculate MSE
errors = test_class - pred
errors = errors**2
print(f"Scikit learn L2 MSE: {sum(errors)/len(errors)}")


#fit the model using scikit learn linear regression with L1 regularization
regL = Lasso(alpha=0.0001)
regL.fit(train_data, train_class)

#predict
pred = regL.predict(test_data).reshape(-1, 1)
print(pred.shape)
#calculate MSE
errors = test_class - pred
errors = errors**2
print(f"Scikit learn L1 MSE: {sum(errors)/len(errors)}")


#load winequality white wine data
data = pd.read_csv("wine+quality/winequality-white.csv", na_values=["?"], header=0, delimiter=";")

#print(data)

data = data.to_numpy()

target = data[:, -1:]
data = data[:, :-1]

#train test split
train_data = data[:int(data.shape[0]*0.8), :]
test_data = data[int(data.shape[0]*0.8):, :]
train_class = target[:int(data.shape[0]*0.8)]
test_class = target[int(data.shape[0]*0.8):]

rows = train_data.shape[0]
features = train_data.shape[1]

#implement linear regression with L2 regularization using gradient descent
#initialize weights
weights = np.random.rand(features, 1)
#initialize regularization parameter
alpha = 1
#initialize learning rate
eta = 0.0000011
#initialize number of iterations
n = 150

#initialize error list
errors1 = []
gradients = []

#gradient descent
for i in range(n):
    #calculate predictions
    pred = np.dot(train_data, weights)
    #print(np.argmax(train_data))
    gradient = 2 * alpha * weights + 2/train_data.shape[0] * np.dot(train_data.T, pred - train_class) #2/train_data.shape[0] to average the gradient
    # update weights
    weights = weights - eta * gradient

    #calculate error
    gradients.append(np.linalg.norm(gradient))
    error = sum((train_class - pred)**2)/train_data.shape[0]
    errors1.append(error)

    #stop if error is small enough
    if error < 0.0001:
        break


#plot error
plt.plot(errors1)
plt.yscale("log")
plt.show()

#plot gradient
plt.plot(gradients)
#plt.yscale("log")
plt.show()

#predict
pred = np.dot(test_data, weights)
print(pred.shape)
#calculate MSE
errors = test_class - pred
errors = errors**2
print(f"Gradient descent L2 MSE: {sum(errors)/len(errors)}")
print(f"Gradient descent L2 MSE minimum gradient: {min(gradients)}")
print(f"Gradient descent L2 MSE minimum gradien index: {np.argmin(gradients)}")


#Stohastic gradient descent
#initialize weights
weights = np.random.rand(features, 1)
#initialize regularization parameter
alpha = 1
#initialize learning rate
eta = 0.0000011
#initialize number of iterations
n = 10

#initialize error list
errors2 = []
gradients = []

#stohastic gradient descent for ridge regression
for i in range(n):
    print(i)
    for j in range(rows):
        #calculate gradient
        gradient = np.dot(train_data[j, :].reshape(1, -1).T, (np.dot(train_data[j, :].reshape(1, -1), weights) - train_class[j])) + alpha*weights
        #update weights
        weights = weights - eta*gradient
        #calculate error
        pred = np.dot(test_data, weights)
        errors2.append(sum((test_class - pred)**2)/len(pred))
        gradients.append(sum(gradient**2)/len(gradient))






#plot error
plt.plot(errors2)
plt.yscale("log")
plt.show()

#plot gradient
plt.plot(gradients)
#plt.yscale("log")
plt.show()

#predict
pred = np.dot(test_data, weights)
print(pred.shape)
#calculate MSE
errors = test_class - pred
errors = errors**2
print(f"Stohastic gradient descent L2 MSE: {sum(errors)/len(errors)}")
print(f"Stohastic gradient descent L2 MSE minimum gradient: {min(gradients)}")
print(f"Stohastic gradient descent L2 MSE minimum gradien index: {np.argmin(gradients)}")




