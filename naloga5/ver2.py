import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def fit(self, X, y):
        # Initialize gamma values as log odds
        positive_class = np.sum(y == 1)
        negative_class = np.sum(y == 0)
        gamma = np.log(positive_class / negative_class)

        #expand gamma to the size of X
        gamma = np.full(X.shape[0], gamma)

        for _ in range(self.n_estimators):
            # Calculate initial probabilities using the sigmoid function
            probabilities = self.sigmoid(gamma)

            # Calculate residuals
            residuals = y - probabilities

            # Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(max_depth=5)
            tree.fit(X, residuals)

            # Calculate intermediary values for each leaf
            leaf_values = tree.apply(X)
            #print(leaf_values)
            intermediary_values = np.zeros(tree.tree_.node_count, dtype=float)

            for leaf in np.unique(leaf_values):
                #find all cases that go to leaf
                leaf_samples = np.where(leaf_values == leaf)[0]

                #sum all residuals for cases that go to leaf
                leaf_sum_residuals = np.sum(residuals[leaf_samples])

                #sum all probabilities for cases that go to leaf
                leaf_sum_weights = np.sum(probabilities[leaf_samples] * (1 - probabilities[leaf_samples]))

                #calculate intermediary value for leaf -> new gamma value
                intermediary_values[leaf] = leaf_sum_residuals / leaf_sum_weights if leaf_sum_weights != 0 else 0

            # Update gamma values
            gamma += self.learning_rate * intermediary_values[leaf_values]

            # Store the tree in the list of models
            self.models.append(tree)

    def predict_proba(self, X):
        # Initialize gamma values as log odds
        gamma = np.zeros(X.shape[0], dtype=float)

        # Sum up predictions from all trees
        for tree in self.models:
            leaf_values = tree.apply(X)
            gamma += self.learning_rate * np.take(tree.tree_.value[:, 0, 0], leaf_values)

        # Convert gamma values to probabilities using the sigmoid function
        probabilities = self.sigmoid(gamma)

        # Return probabilities for class 1 (assuming binary classification)
        return np.vstack((1 - probabilities, probabilities)).T

    def predict(self, X, threshold=0.5):
        # Make probability predictions
        probabilities = self.predict_proba(X)

        # Convert probabilities to binary predictions based on the threshold
        binary_predictions = (probabilities[:, 1] >= threshold).astype(int)

        return binary_predictions

# Example usage:
# Assuming you have your training data in X_train and labels in y_train
# and test data in X_test

# Generate synthetic data for testing
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Fit the model
gb_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



#load data from the wine_quality csv
data = pd.read_csv("wine_quality.csv", na_values=["?"], header=0, delimiter=",")
data = data.dropna()

#encode the alcohol_level column as 0 / 1 instead of low / high
data["alcohol_level"] = data["alcohol_level"].apply(lambda x: 0 if x == "low" else 1)

data = data.to_numpy()

#shuffle the data
np.random.shuffle(data)

#split last column as the target variable
target = data[:, -1]

#set the rest of the data as the features, except for first column which is index, therefore it should be drpped
features = data[:, 1:-1]

#train test split
train_data = features[:int(features.shape[0]*0.8), :]
test_data = features[int(features.shape[0]*0.8):, :]
train_class = target[:int(target.shape[0]*0.8)]
test_class = target[int(target.shape[0]*0.8):]

#fit the model and make predictions
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb_clf.fit(train_data, train_class)
y_pred = gb_clf.predict(test_data)

#evaluate accuracy
accuracy = accuracy_score(test_class, y_pred)
print(f"Accuracy: {accuracy}")

print()
print("GRID SEARCH")

#test different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
best_accuracy = 0
best_learning_rate = 0
for learning_rate in learning_rates:
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate)
    gb_clf.fit(train_data, train_class)
    y_pred = gb_clf.predict(test_data)
    accuracy = accuracy_score(test_class, y_pred)
    print(f"Accuracy for learning rate {learning_rate}: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_learning_rate = learning_rate

#test different number of estimators
n_estimators = [10, 50, 100, 200, 500, 1000]
best_accuracy = 0
best_n_estimators = 0
for n_estimator in n_estimators:
    gp_cfl = GradientBoostingClassifier(n_estimator, best_learning_rate)
    gb_clf.fit(train_data, train_class)
    y_pred = gb_clf.predict(test_data)
    accuracy = accuracy_score(test_class, y_pred)
    print(f"Accuracy for number of trees {n_estimator}, using the best learning rate from before ({best_learning_rate}): {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_estimators = n_estimator


#grid search for best combination of parameters
best_accuracy = 0
best_learning_rate = 0
best_n_estimators = 0
for learning_rate in learning_rates:
    for n_estimator in n_estimators:
        gb_clf = GradientBoostingClassifier(n_estimator, learning_rate)
        gb_clf.fit(train_data, train_class)
        y_pred = gb_clf.predict(test_data)
        accuracy = accuracy_score(test_class, y_pred)
        print(f"Accuracy for number of trees {n_estimator}, learning rate {learning_rate}: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate = learning_rate
            best_n_estimators = n_estimator

print(f"Best accuracy: {best_accuracy}, best learning rate: {best_learning_rate}, best n_estimators: {best_n_estimators}")
print()

#calculate results on the test set
gb_clf = GradientBoostingClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate)
gb_clf.fit(train_data, train_class)
y_pred = gb_clf.predict(test_data)
accuracy = accuracy_score(test_class, y_pred)
print(f"Accuracy of my gradient boost implementation with best parameters on the test set: {accuracy}")

#compare results to GradientBoostingClassifier from sklearn frok scikit-learn

#initate the sklearn gradient boosting classifier
gb_clf_sklearn = SklearnGradientBoostingClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate)

#fit the model and make predictions
gb_clf_sklearn.fit(train_data, train_class)
y_pred = gb_clf_sklearn.predict(test_data)

#evaluate accuracy
accuracy = accuracy_score(test_class, y_pred)
print(f"Accuracy for sklearn gradient boosting clasifier: {accuracy}")

print()
print("CV")
#cross validation for gradient boositng classification

def cv_gb_cl(train_data, train_class, n_estimators, learning_rate, k=5, model=GradientBoostingClassifier):
    #split the data into k folds
    folds = np.array_split(train_data, k)
    folds_class = np.array_split(train_class, k)

    #initiate the list of accuracies
    accuracies = []

    #for each fold, train the model on the rest of the folds and test on the current fold
    for i in range(k):
        #initiate the model
        gb_clf = model(n_estimators=n_estimators, learning_rate=learning_rate)

        #initiate the training data and class
        train_data = np.concatenate(folds[:i] + folds[i+1:])
        train_class = np.concatenate(folds_class[:i] + folds_class[i+1:])

        #fit the model
        gb_clf.fit(train_data, train_class)

        #make predictions
        y_pred = gb_clf.predict(folds[i])

        #evaluate accuracy
        accuracy = accuracy_score(folds_class[i], y_pred)
        accuracies.append(accuracy)

    #return the mean accuracy
    return np.mean(accuracies)

print(f"CV score for my implementation of gradient boosting classifier: {cv_gb_cl(train_data, train_class, best_n_estimators, best_learning_rate)}")
print(f"CV score for sklearn gradient boosting classifier: {cv_gb_cl(train_data, train_class, best_n_estimators, best_learning_rate, model=SklearnGradientBoostingClassifier)}")

#also test models XGBoost, LightGBM and Catboost

#XGBoost
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate)

print()
print(f"CV score for XGBoost: {cv_gb_cl(train_data, train_class, best_n_estimators, best_learning_rate, model=XGBClassifier)}")
print()

#LightGBM
from lightgbm import LGBMClassifier
lgbm_clf = LGBMClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate)
print()
print(f"CV score for LightGBM: {cv_gb_cl(train_data, train_class, best_n_estimators, best_learning_rate, model=LGBMClassifier)}")
print()
#Catboost
from catboost import CatBoostClassifier
cat_clf = CatBoostClassifier(n_estimators=best_n_estimators, learning_rate=best_learning_rate, verbose=False)
print()
print(f"CV score for Catboost: {cv_gb_cl(train_data, train_class, best_n_estimators, best_learning_rate, model=CatBoostClassifier)}")
print()
