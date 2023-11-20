import pandas as pd
import numpy as np

#Load data

data = pd.read_csv("data/House_price.csv", na_values=["?"], header=0, delimiter=",")
data = data.to_numpy()

target = data[:, 0]
data = data[:, 1:]

#train test split
train_data = data[:int(data.shape[0]*0.8), :]
test_data = data[int(data.shape[0]*0.8):, :]
train_class = target[:int(data.shape[0]*0.8)]
test_class = target[int(data.shape[0]*0.8):]

rows = train_data.shape[0]
features = train_data.shape[1]

#Regression tree classes

class Node:
    def __init__(self, feature, value, left, right, indices=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.indices = indices

    def isLeaf(self):
        return self.left == None and self.right == None


class RegressionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth):
        if (depth >= self.max_depth
                or self.n_samples < self.min_samples_split):
            return True
        return False

    def _build_tree(self, data, target, depth=0):
        self.n_samples, self.n_features = data.shape
        #print(data.shape)
        if(self._is_finished(depth)):
            average_label = np.average(target)
            return Node(value=average_label, feature=None, left=None, right=None)

        best_split = self._best_split(data, target, np.arange(data.shape[0])) #check if this okay with best split
        left = data[best_split["left_indices"]]
        right = data[best_split["right_indices"]]
        left_target = target[best_split["left_indices"]]
        right_target = target[best_split["right_indices"]]

        if(len(best_split["left_indices"]) == 0 or len(best_split["right_indices"]) == 0):
            average_label = np.average(target)
            return Node(value=average_label, feature=None, left=None, right=None)

        right_child = self._build_tree(right, right_target, depth+1)
        left_child = self._build_tree(left, left_target, depth+1)

        return Node(feature=best_split["feature_index"], value=best_split["feature_value"], left=left_child, right=right_child)


    def _fit(self, data, target):
        self.root = self._build_tree(data, target)

    def _avg_mse(self, target, value):
        #print(target.shape)
        return np.average((target - value)**2)

    """def _create_split(self, data, index):
        left = data[data[:, index] <= data[index], :]
        right = data[data[:, index] > data[index], :]

        return left, right"""

    def _create_split(self, data, sample, index):
        left_indices = np.where(data[:, index] < data[sample, index])
        right_indices = np.where(data[:, index] >= data[sample, index])

        return left_indices, right_indices

    def _best_split(self, data, target, data_indices): #Double check indices, something wrong
        best_mse_score = np.inf
        best_split = {}

        for i in range(data.shape[1]): #iterate over features
            for j in data_indices: #iterate over samples
                left_indices, right_indices = self._create_split(data, j, i)

                left_target = target[left_indices]
                right_target = target[right_indices]

                left_mse_score = self._avg_mse(left_target, target[j])
                right_mse_score = self._avg_mse(right_target, target[j])

                mse_score = (left_mse_score + right_mse_score) / 2

                if mse_score < best_mse_score:
                    best_mse_score = mse_score
                    best_split = {"feature_index": i,
                                  "feature_value": data[j, i],
                                  "left_indices": left_indices,
                                  "right_indices": right_indices}
        print(target)
        print("**********")
        print(left_target, left_indices)
        print("--------------")
        print(right_target, right_indices)
        print("=============")
        return best_split

    def _traverse(self, x, node):
        if node.isLeaf():
            return node.value

        if x[node.feature] < node.value:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)



dummyData = np.random.randint(0, 10, (10, 3))
dummyTarget = np.random.randint(0, 10, (10, 1))



testTree = RegressionTree()
testTree._fit(dummyData, dummyTarget)


