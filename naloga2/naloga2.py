import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_fwf("communities+and+crime/communities.names")
print(data)

data = pd.read_csv("communities+and+crime/communities.data", na_values=["?"], header=None)
data.drop(list(range(5)), axis=1, inplace=True)
data.interpolate(axis=1, inplace=True)
print(data)

print(data.to_numpy())

data = data.to_numpy()

target = data[:, -1:]

data = data[:, :-1]

print(target)
print(data)
print(data.shape)

reg = LinearRegression()


#train test split
train_data = data[:1500, :]
test_data = data[1500:, :]

train_class = target[:1500]
test_class = target[1500:]

rows = train_data.shape[0]
features = train_data.shape[1]

#cross validation
"""
mses = []
n = 5
for i in range(n):
    cv_test = train_data[i*int(rows/n) : (i+1)*int(rows/n), :]
    cv_train = np.delete(train_data, slice(i*int(rows/n), (i+1)*int(rows/n)), axis=0)
    cv_target_test = train_class[i*int(rows/n) : (i+1)*int(rows/n), :]
    cv_target_train = np.delete(train_class, slice(i*int(rows/n), (i+1)*int(rows/n)), axis=0)
    print(cv_train.shape)
    print(cv_target_train.shape)
    reg.fit(cv_train, cv_target_train)
    pred = reg.predict(cv_test)
    errors = cv_target_test - pred
    errors = errors**2
    print(sum(errors)/len(errors))
    mses.append(sum(errors)/len(errors))

print(f"Avg CV MSE: {sum(mses)/len(mses)}")
"""


#leave-one-out
"""
loo_mse = []
for i in range(rows):
    loo_test = train_data[i]
    loo_train = np.delete(train_data, i, axis=0)
    loo_target_test = train_class[i]
    loo_target_train = np.delete(train_class, i, axis=0)

    reg.fit(loo_train, loo_target_train)
    pred = reg.predict(loo_test.reshape(1, -1))

    errors = loo_target_test - pred
    errors = errors**2
    loo_mse.append(sum(errors)/len(errors))

print(f"Avg LOO MSE: {sum(loo_mse)/len(loo_mse)}")
"""

def cv(train_data, train_class, n=5):
    mses = []
    for i in range(n):
        cv_test = train_data[i * int(rows / n): (i + 1) * int(rows / n), :]
        cv_train = np.delete(train_data, slice(i * int(rows / n), (i + 1) * int(rows / n)), axis=0)
        cv_target_test = train_class[i * int(rows / n): (i + 1) * int(rows / n), :]
        cv_target_train = np.delete(train_class, slice(i * int(rows / n), (i + 1) * int(rows / n)), axis=0)
        #print(cv_train.shape)
        #print(cv_target_train.shape)
        reg.fit(cv_train, cv_target_train)
        pred = reg.predict(cv_test)
        errors = cv_target_test - pred
        errors = errors ** 2
        #print(sum(errors) / len(errors))
        mses.append(sum(errors) / len(errors))
    return sum(mses)/len(mses)

"""
#Forward feature selection
features_out = list(range(features))
data_ffs = None
target_ffs = None

ffs_mses = []
ffs_feature_selections = []

for i in range(features):
    print(i)
    print(features_out)
    best_feat = -1
    best_mse = np.inf
    for j in features_out:
        if(data_ffs is None):
            ffs_split = train_data[:, j].reshape(-1, 1)
            mse = cv(ffs_split, train_class)
            if(mse < best_mse):
                best_mse = mse
                best_feat = j
        else:
            ffs_split = np.c_[data_ffs, train_data[:, j].reshape(-1, 1)]
            mse = cv(ffs_split, train_class)
            if (mse < best_mse):
                best_mse = mse
                best_feat = j
    ffs_mses.extend(best_mse)
    features_out.remove(best_feat)
    ffs_feature_selections.append(best_feat)
    if(data_ffs is None):
        data_ffs = train_data[:, best_feat].reshape(-1, 1)
    else:
        data_ffs = np.c_[data_ffs, train_data[:, best_feat].reshape(-1, 1)]

print(ffs_mses)
print(data_ffs.shape)

np.save("ffs_data.npy", data_ffs)
np.save("ffs_mses.npy", ffs_mses)
np.save("ffs_feature_selections.npy", ffs_feature_selections)
"""


data_ffs = np.load("ffs_data.npy")
ffs_mses = np.load("ffs_mses.npy")
ffs_feature_selections = np.load("ffs_feature_selections.npy")
plt.plot(ffs_mses)
plt.show()
print(ffs_mses)

print(np.argmin(ffs_mses))

split_index = np.argmin(ffs_mses)


#feature selection according to optimal feature set
data_ffs = data_ffs[:, :split_index+1]

test_data_ffs = None
for i in ffs_feature_selections[:split_index+1]:
    if test_data_ffs is None:
        test_data_ffs = test_data[:, i].reshape(-1, 1)
    else:
        test_data_ffs = np.c_[test_data_ffs, test_data[:, i].reshape(-1, 1)]
reg.fit(data_ffs, train_class)
pred = reg.predict(test_data_ffs)
errors = test_class - pred
errors = errors**2
print(f"Test MSE: {sum(errors)/len(errors)}")

#MSE without feature selection
reg.fit(train_data, train_class)
pred = reg.predict(test_data)
errors = test_class - pred
errors = errors**2
print(f"Test MSE whitout feature selection: {sum(errors)/len(errors)}")

#Implement the bootstrap method and apply it to the train set to generate 1000 different train sets and train 1000 different linear models

#Bootstrap

def bootstrap(train_data, train_class, n=1000):
    mses = []
    rows = train_data.shape[0]
    features = train_data.shape[1]
    for i in range(n):
        bootstrap_train = np.zeros((rows, features))
        bootstrap_target_train = np.zeros((rows, 1))
        for j in range(rows):
            index = np.random.randint(0, rows)
            bootstrap_train[j] = train_data[index]
            bootstrap_target_train[j] = train_class[index]
        reg.fit(bootstrap_train, bootstrap_target_train)
        pred = reg.predict(test_data)
        errors = test_class - pred
        errors = errors ** 2
        mses.append(sum(errors) / len(errors))
    return mses

bootstrap_mses = bootstrap(train_data, train_class)
print(f"Bootstrap MSE: {sum(bootstrap_mses)/len(bootstrap_mses)}")

#Use the bootstrapped results to assess the confidence intervals of the results of the linear model.

bootstrap_mses.sort()
print(bootstrap_mses[int(0.025*len(bootstrap_mses))])
print(bootstrap_mses[int(0.975*len(bootstrap_mses))])

#plot a histogram of bootstrapped MSEs
bootstrap_mses_plot = bootstrap_mses.copy()
#round all values to 3 decimal places
for i in range(len(bootstrap_mses_plot)):
    bootstrap_mses_plot[i] = round(bootstrap_mses_plot[i][0], 4)
plt.hist(bootstrap_mses_plot, bins=1000, rwidth=1)
plt.show()


