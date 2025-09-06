import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):

        # calculate and remove mean
        self.mean = X.mean(axis=0)
        X = X - self.mean
        
        # create covariance
        cov = np.cov(X.T)
        
        # create eigen vectors and eigen values
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # create principal components
        eigenvectors    = eigenvectors
        idxs            = np.argsort(eigenvalues[::-1])
        eigenvalues     = eigenvalues[idxs]
        eigenvectors    = eigenvectors[idxs]
        self.components = eigenvectors[:, :self.n_components]
        print(self.components.shape)

        # transform the data
        return np.dot(X, self.components)

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)