import numpy as np


class KMeans:
    
    def __init__(self, n_clusters=3, max_iter=1_000):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.centers    = None

    def euclidian(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def create_centers(self, X):
        n_samples, n_features = X.shape
        self.centers = np.zeros(shape=(self.n_clusters, n_features))
        # create starter centers
        mins, maxs = np.min(X, axis=0), np.max(X, axis=0)
        bounds = [[lb, ub] for lb, ub in zip(mins, maxs)]
        for j in range(self.n_clusters):
            center = np.zeros(n_features)
            for i, bound in enumerate(bounds):
                xi_min, xi_max = bound
                center[i] = np.random.uniform(low=xi_min, high=xi_max)
            self.centers[j,:] = center

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize random centers
        self.create_centers(X)

        # until max iter or convergence
        for _ in range(self.max_iter):
            # sample to centers distances
            center_map = np.zeros(n_samples)
            for i in range(n_samples):
                center_map[i] = np.argmin([self.euclidian(X[i], c) for c in self.centers])
            # recompute new centers
            new_centers = np.zeros(shape=(self.n_clusters, n_features))
            for i in range(self.n_clusters):
                new_centers[i] = np.mean(X[center_map == i], axis=0)

            # check if centers have shifted ( todo )
            if np.allclose(self.centers, new_centers): 
                self.centers = new_centers
                break

            # update centers
            self.centers = new_centers

    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = np.argmin([self.euclidian(X[i], c) for c in self.centers])
        return y_pred
