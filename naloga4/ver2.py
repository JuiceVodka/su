import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("data/House_price.csv", na_values=["?"], header=0, delimiter=",")
#remove NaN values
data = data.dropna()

#change "yes" and "no" to 1 and 0
data['bus_ter'] = (data['bus_ter'] =='YES').astype(int)
data['airport'] = (data['airport'] =='YES').astype(int)
#print(data['bus_ter'])
#print(data)
#df['viz'] = (df['viz'] !='n').astype(int)

#remove column waterbody
data = data.drop("waterbody", axis=1)

data = data.to_numpy()

target = data[:, 0]
data = data[:, 1:]

#train test split
train_data = data[:int(data.shape[0]*0.8), :]
test_data = data[int(data.shape[0]*0.8):, :]
train_class = target[:int(data.shape[0]*0.8)]
test_class = target[int(data.shape[0]*0.8):]





class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y))**2)

    def best_split(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None  # Stop splitting if the number of samples is below the threshold

        overall_mse = self.mse(y)

        best_mse = float('inf')
        best_index = None
        best_value = None

        for i in range(n):
            unique_values = np.unique(X[:, i])
            for value in unique_values:
                left_mask = X[:, i] <= value
                right_mask = ~left_mask

                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                left_mse = self.mse(y[left_mask])
                right_mse = self.mse(y[right_mask])

                weighted_mse = (len(y[left_mask]) / m) * left_mse + (len(y[right_mask]) / m) * right_mse

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_index = i
                    best_value = value

        if best_mse < overall_mse:
            return best_index, best_value
        else:
            return None, None  # Stop splitting if the best split does not reduce MSE

    def grow_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        best_index, best_value = self.best_split(X, y)

        if best_index is not None:
            left_mask = X[:, best_index] <= best_value
            right_mask = ~left_mask

            left_subtree = self.grow_tree(X[left_mask, :], y[left_mask], depth + 1)
            right_subtree = self.grow_tree(X[right_mask, :], y[right_mask], depth + 1)

            return (best_index, best_value, left_subtree, right_subtree)
        else:
            return np.mean(y)

    def fit(self, X, y):
        self.tree = self.grow_tree(X, y)

    def predict_single(self, x, tree): #predicts one sample
        if isinstance(tree, np.float64):
            return tree  # Leaf node, return the predicted value
        else:
            index, value, left_subtree, right_subtree = tree
            if x[index] <= value:
                return self.predict_single(x, left_subtree)
            else:
                return self.predict_single(x, right_subtree)

    def predict(self, X):
        if self.tree is None:
            print("Model not trained yet. Call fit() before predict().")
            return None

        return np.array([self.predict_single(x, self.tree) for x in X])

# Unit case test to see if it runs
# Assuming X_train and y_train are your training data
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([2, 4, 6, 8])

# Create and fit the regression tree
reg_tree = RegressionTree(max_depth=3, min_samples_split=2)
reg_tree.fit(X_train, y_train)

# Example prediction
X_test = np.array([[2, 3], [6, 7]])
predictions = reg_tree.predict(X_test)
print(predictions)

#test for real data
reg_tree = RegressionTree(max_depth=3, min_samples_split=2)
reg_tree.fit(train_data, train_class)
predictions = reg_tree.predict(test_data)
#print(predictions)

#calculate mse
mse = np.mean((predictions - test_class)**2)
print(mse)


def cross_validate_regression_tree(X, y, n_folds=5, max_depth=None, min_samples_split=2):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and fit the regression tree
        reg_tree = RegressionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        reg_tree.fit(X_train, y_train)

        # Predict on the test set
        y_pred = reg_tree.predict(X_test)

        # Evaluate the model using Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

max_cv_score = 1000
max_std = 1000
best_depth = 1000
for depth in range(1, 10):
    cv_score, cv_std = cross_validate_regression_tree(data, target, n_folds=5, max_depth=depth, min_samples_split=2)
    if cv_score < max_cv_score:
        max_cv_score = cv_score
        best_depth = depth
        max_std = cv_std
print(f"Best depth: {best_depth}, MSE: {max_cv_score}, std_dev: {max_std}")

cv_score = cross_validate_regression_tree(data, target, n_folds=5, max_depth=6, min_samples_split=2)
print(cv_score)

#compare with sci-kit learn regression tree
#print(train_data)
#print(train_class)
reg_tree = DecisionTreeRegressor(max_depth=6, min_samples_split=2)
reg_tree.fit(train_data, train_class)
predictions = reg_tree.predict(test_data)
mse = np.mean((predictions - test_class)**2)
print(mse)


#cross validation for sci-kit learn regression tree
def cross_validate_regression_tree_scikit(X, y, n_folds=5, max_depth=None, min_samples_split=2):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and fit the regression tree
        reg_tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        reg_tree.fit(X_train, y_train)

        # Predict on the test set
        y_pred = reg_tree.predict(X_test)

        # Evaluate the model using Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

max_cv_score = 1000
max_std = 1000
best_depth_scikit = 1000
for depth in range(1, 10):
    cv_score, cv_std = cross_validate_regression_tree_scikit(data, target, n_folds=5, max_depth=depth, min_samples_split=2)
    if cv_score < max_cv_score:
        max_cv_score = cv_score
        max_std = cv_std
        best_depth_scikit = depth

print(f"Best depth for scikit learn: {best_depth}, MSE for scikit learn: {max_cv_score}, std_dev for scikit learn: {max_std}")

#use the best depth parameters to train the model and predict on the test data
my_tree = RegressionTree(max_depth=best_depth, min_samples_split=2)
cv_tree = DecisionTreeRegressor(max_depth=best_depth_scikit, min_samples_split=2)

my_tree.fit(train_data, train_class)
cv_tree.fit(train_data, train_class)

my_predictions = my_tree.predict(test_data)
cv_predictions = cv_tree.predict(test_data)

my_mse = np.mean((my_predictions - test_class)**2)
cv_mse = np.mean((cv_predictions - test_class)**2)

print(f"MSE on test set for my tree: {my_mse}")
print(f"MSE on test set for scikit learn tree: {cv_mse}")