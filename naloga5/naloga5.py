from sklearn import tree
import numpy as np
import pandas as pd

#Tole je chatgpt izpluniu
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Initialize the model with the log odds (gamma values)
        initial_gamma = np.log(np.sum(y == 1) / np.sum(y == 0))
        initial_gamma = np.full(X.shape[0], initial_gamma)
        self.models.append(initial_gamma)
        print(X.shape)
        print(y.shape)
        print(initial_gamma.shape)
        print(initial_gamma)

        gamma_values = np.zeros((X.shape[0], self.n_estimators+1))
        gamma_values[:, 0] = initial_gamma

        for i in range(self.n_estimators):
            # Calculate initial probabilities using the sigmoid function
            initial_probabilities = self.sigmoid(self.models[-1])
            #print(initial_probabilities.shape)
            # Calculate residuals
            residuals = y - initial_probabilities

            # Fit a weak learner (DecisionTreeRegressor) to the residuals
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)

            # Calculate intermediary values for each leaf
            leaf_values = tree.apply(X)
            print(leaf_values.shape)

            leaf_sum = np.bincount(leaf_values, weights=residuals)
            leaf_count = np.bincount(leaf_values)
            leaf_values = leaf_sum / (leaf_count + 1e-8)  # Avoid division by zero

            # Update the model with the new gamma values
            self.models.append(self.models[-1] + self.learning_rate * leaf_values)

    def predict_proba(self, X):
        # Make predictions by summing the predictions of all weak learners and applying the sigmoid function
        predictions = sum(model.predict(X) for model in self.models[1:])
        return self.sigmoid(predictions)

    def predict(self, X, threshold=0.5):
        # Make binary predictions based on a threshold
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Generate synthetic data for testing
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")