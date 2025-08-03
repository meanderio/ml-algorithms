import numpy as     np
class KNeighborsClassifier:
    def __init__(self, k=3):
        self._k = k
        self._X = None
        self._y = None
    
    def _euclidian_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def fit(self, X, y):
        self._X = X
        self._y = y

    def _predict_nn(self, dists):
        idx = np.argpartition(dists, self._k)
        knn = self._y[idx[:self._k]]
        v,c = np.unique(knn, return_counts=True)
        return v[np.argmax(c)]
    
    def _compute_dists(self, x):
        n_samples = self._X.shape[0]
        dists = np.zeros(n_samples)
        # compare one test to all train
        for j in range(n_samples):
            # compute distance
            dists[j] = self._euclidian_distance(x, self._X[j])
        return dists

    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples)
        # iterate through each test sample
        for i in range(n_samples):
            # compute dist to all train
            dists     = self._compute_dists(X[i])
            # predict class from k nearest
            y_pred[i] = self._predict_nn(dists)
        return y_pred
    
    def score(self, y_true, y_pred):
        matches = [1 if a==b else 0 for a,b in zip(y_true, y_pred)]
        return float(matches) / len(y_true)
