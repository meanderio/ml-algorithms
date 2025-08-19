import numpy   as np
import seaborn as sns
from matplotlib import pyplot as plt


class KMeans:
    
    def __init__(self, n_clusters=3, max_iter=1_000, plot=False):
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.centers    = None
        self.plot       = plot

    def euclidian(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def initialize_centers(self, X):
        n_samples, n_features = X.shape
        center_idxs = np.random.choice([i for i in range(n_samples)], size=self.n_clusters, replace=False)
        self.centers = X[center_idxs]
    
    def fit(self, X):
        n_samples, n_features = X.shape
        # initialize random centers
        self.initialize_centers(X)

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

            if self.plot: self.plot_step(X, center_map, new_centers)
            
            # check if centers have shifted ( todo )
            if np.allclose(self.centers, new_centers): 
                self.centers = new_centers
                break

            # update centers
            self.centers = new_centers

    def plot_step(self, X, y, centers):
        sns.scatterplot(
            x = [x[0] for x in X],
            y = [x[1] for x in X],
            hue     = y,
            palette = "deep",
            legend  = None
        )
        sns.scatterplot(
            x = [x[0] for x in centers],
            y = [x[1] for x in centers],
            color=".1", marker="+",
            palette = "deep",
            legend  = None
        )

        plt.xlabel("x1")
        plt.ylabel("x1")
        plt.show()

    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = np.argmin([self.euclidian(X[i], c) for c in self.centers])
        return y_pred, self.centers
