import sys
import numpy as     np
from   scipy import stats
sys.path.append("../decision-tree")

from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, min_samples=2, max_depth=4):
        self.n_trees     = n_trees
        self.min_samples = min_samples
        self.max_depth   = max_depth
        self.trees       = []

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples = self.min_samples,
                max_depth   = self.max_depth
            )
            X_b, y_b = self._bootstrap(X, y)
            tree.fit(X_b, y_b)
            self.trees.append(tree)
        return None

    def predict(self, X):
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(X))
        P = np.matrix(tree_predictions)
        #print(P)
        y_pred, counts = stats.mode(P, axis=0)
        return y_pred