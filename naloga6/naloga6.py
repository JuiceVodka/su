import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data.csv", na_values=["?"], delimiter=",", header=0)
print(data)
data.drop("Unnamed: 32", axis=1, inplace=True)
data_np = data.to_numpy()
data_np = data_np[:, :-1]
np.random.shuffle(data_np)
print(data_np)
print(data)

target = data_np[:, 1]
data_np = data_np[:, 2:]

#train test split
train_data = data_np[:350, :]
test_data = data_np[350:, :]
train_class = target[:350]
test_class = target[350:]


clf = svm.SVC()
clf.fit(train_data, train_class)

pred = clf.predict(test_data)
print(pred)

#results are either B or M, so we can calculate accuracy as follows:
test_class_encoded = np.where(test_class == "B", 0, 1)
print(test_class_encoded)
predictions_encoded = np.where(pred == "B", 0, 1)

print(f"Accuracy with default kernel: {sum(test_class_encoded == predictions_encoded)/len(test_class_encoded)}")

#With kernels
#kernels: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

def scikit_svm(kernel, regularization, sigma, data_train, data_test, target_train, target_test):
    clf = svm.SVC(kernel=kernel, C=regularization, gamma=sigma)
    clf.fit(data_train, target_train)
    pred = clf.predict(data_test)
    test_class_encoded = np.where(target_test == "B", 0, 1)
    predictions_encoded = np.where(pred == "B", 0, 1)
    return sum(test_class_encoded == predictions_encoded)/len(test_class_encoded)

#grid search for best regularization parameter (i), kernel, and sigma parameter (j)
for i in [0.01, 0.1, 0.5, 1, 2, 5, 10]:
    print(f"Accuracy with linear kernel and regularization parameter {i}: {scikit_svm('linear', i, 'scale', train_data, test_data, train_class, test_class)}")
    print(f"Accuracy with poly kernel and regularization parameter {i}: {scikit_svm('poly', i, 'scale', train_data, test_data, train_class, test_class)}")
    for j in ["scale", "auto"]:
        print(f"Accuracy with rbf kernel and regularization parameter {i} and sigma parameter {j}: {scikit_svm('rbf', i, j, train_data, test_data, train_class, test_class)}")
    print(f"Accuracy with sigmoid kernel and regularization parameter {i}: {scikit_svm('sigmoid', i, 'scale', train_data, test_data, train_class, test_class)}")
    print()



#implement kernel regression to model data2

data = pd.read_csv("data2.csv", na_values=["?"], delimiter=",", header=0)

data_np = data.to_numpy()

#standardize data
#you should standardise od train and test separately
class_std = np.std(data_np[:, -1])
for i in range(data_np.shape[1]):
    data_np[:, i] = (data_np[:, i] - np.mean(data_np[:, i])) / np.std(data_np[:, i])


data_np = data_np[:, 1:]
np.random.shuffle(data_np)
target = data_np[:, -1]
data_np = data_np[:, 0]


#train test split
train_data = data_np[:120]
test_data = data_np[120:]
train_class = target[:120]
test_class = target[120:]

#implement kernel functions
def linear_kernel(x, y, dummy=None):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def exponential_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y) / sigma ** 2)


def kernel_regression(kernel, sigma, regularization, data_train, data_test, target_train, target_test):
    #calculate kernel matrix
    kernel_matrix = np.zeros((len(data_train), len(data_train)))
    for i in range(len(data_train)):
        for j in range(len(data_train)):
            kernel_matrix[i, j] = kernel(data_train[i], data_train[j], sigma)

    #calculate alpha
    weights = np.dot(np.linalg.inv(kernel_matrix + regularization*np.eye(len(data_train))), target_train)

    #calculate predictions
    predictions = np.zeros(len(data_test))
    for i in range(len(data_test)):
        kernel_vec = np.zeros(len(data_train))
        for j in range(len(data_train)):
            kernel_vec[j] = kernel(data_test[i], data_train[j], sigma)
        predictions[i] = np.dot(kernel_vec, weights)

    return predictions

#calculate mean and sigma of class
mean = np.mean(train_class)
sigma = np.std(train_class)
print(f"Mean: {mean}, sigma: {sigma} of class (standardized))")

#calculate predictions
predictions = kernel_regression(linear_kernel, 1, 1, train_data, test_data, train_class, test_class)
#print(predictions)
#calculate MSE
print(f"MSE for linear kernel: {np.mean((predictions - test_class)**2)}")
print(f"MSE for linear kernel corrected with std_dev of data before standardization: {np.mean((predictions - test_class)**2) * (class_std ** 2)}")


predictions = kernel_regression(polynomial_kernel, 3, 1, train_data, test_data, train_class, test_class)
#print(predictions)
#calculate MSE
print(f"MSE for polynomial kernel: {np.mean((predictions - test_class)**2)}")
print(f"MSE for polynomial kernel corrected with std_dev of data before standardization: {np.mean((predictions - test_class)**2) * (class_std ** 2)}")


predictions = kernel_regression(gaussian_kernel, sigma, 1, train_data, test_data, train_class, test_class)
#print(predictions)
#calculate MSE
print(f"MSE for gaussian kernel: {np.mean((predictions - test_class)**2)}")
print(f"MSE for gaussian kernel corrected with std_dev of data before standardization: {np.mean((predictions - test_class)**2) * (class_std ** 2)}")

predictions = kernel_regression(exponential_kernel, 1, 1, train_data, test_data, train_class, test_class)
#print(predictions)
#calculate MSE
print(f"MSE for exponential kernel: {np.mean((predictions - test_class)**2)}")
print(f"MSE for exponential kernel corrected with std_dev of data before standardization: {np.mean((predictions - test_class)**2) * (class_std ** 2)}")

#compare with linear regression with scikit learn
reg = LinearRegression().fit(train_data.reshape(-1, 1), train_class)
print(f"MSE for linear regression with scikit learn: {np.mean((reg.predict(test_data.reshape(-1, 1)) - test_class)**2)}")
print(f"MSE for linear regression with scikit learn corrected with std_dev of data before standardization: {np.mean((reg.predict(test_data.reshape(-1, 1)) - test_class)**2) * (class_std ** 2)}")


#best model is shown to the polynomial kernel with degree 3
#plot predictions and the polynomial regression line
plt.scatter(test_data, test_class, label="test data")
plt.scatter(test_data, predictions, label="predictions")
plt.legend()
plt.show()

#polinom seems to be of higher order; grid search for best degree
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
best_degree = 0
best_mse = 1000000000
for i in degrees:
    predictions = kernel_regression(polynomial_kernel, i, 1, train_data, test_data, train_class, test_class)
    print(f"MSE for polynomial kernel with degree {i}: {np.mean((predictions - test_class)**2)}")
    if np.mean((predictions - test_class)**2) < best_mse:
        best_mse = np.mean((predictions - test_class)**2)
        best_degree = i

print(f"Best degree: {best_degree}, best MSE: {best_mse}")

#plot predictions and the polynomial regression line
predictions = kernel_regression(polynomial_kernel, best_degree, 1, train_data, test_data, train_class, test_class)
plt.title("Polynomial kernel with best degree predictions")
plt.scatter(test_data, test_class, label="test data")
plt.scatter(test_data, predictions, label="predictions")
plt.legend()
plt.show()

#plot residuals vs predictions
plt.scatter(test_class, predictions - test_class)
plt.title("Polynomial kernel with best degree residuals")
plt.xlabel("Class")
plt.ylabel("Residuals")
plt.show()




#test plot for linear regression
predictions = kernel_regression(polynomial_kernel, 1, 1, train_data, test_data, train_class, test_class)
plt.title("Linear kernel predictions")
plt.scatter(test_data, test_class, label="test data")
plt.scatter(test_data, predictions, label="predictions")
plt.legend()
plt.show()

plt.scatter(test_class, predictions - test_class)
plt.title("Linear kernel residuals")
plt.xlabel("Class")
plt.ylabel("Residuals")
plt.show()
#confirm it looks like a polynomial regression

#plot predictions with exponential kernel
predictions = kernel_regression(exponential_kernel, 5, 1, train_data, test_data, train_class, test_class)
plt.title("Exponential kernel predictions")
plt.scatter(test_data, test_class, label="test data")
plt.scatter(test_data, predictions, label="predictions")
plt.legend()
plt.show()

#plot residuals for exponential kernel
plt.scatter(test_class, predictions - test_class)
plt.title("Exponential kernel residuals")
plt.xlabel("Class")
plt.ylabel("Residuals")
plt.show()
#exponential kernel is not the best choice for this data set

#plot predictions with gaussian kernel
predictions = kernel_regression(gaussian_kernel, 1, 1, train_data, test_data, train_class, test_class)
plt.title("Gaussian kernel predictions")
plt.scatter(test_data, test_class, label="test data")
plt.scatter(test_data, predictions, label="predictions")
plt.legend()
plt.show()

#plot residuals for gaussian kernel
plt.scatter(test_class, predictions - test_class)
plt.title("Gaussian kernel residuals")
plt.xlabel("Class")
plt.ylabel("Residuals")
plt.show()
