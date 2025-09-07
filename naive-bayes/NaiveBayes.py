import numpy as np

class NaiveBayes:
    def __init__(self,):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes   = np.unique(y)
        self.n_classes = len(self.classes)
        # setup mean, var, prior for classes
        self.means = np.zeros((self.n_classes, n_features))
        self.vars  = np.zeros((self.n_classes, n_features))
        self.priors = np.zeros(self.n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[ y == c ]
            self.means[idx, :] = np.mean(X_c, axis=0)
            self.vars[idx, :]  = np.var(X_c, axis=0)
            self.priors[idx]   = y[y == c].shape[0] / n_samples

    def gaussian(self, x, u, v):
        return (1 / (np.sqrt(2*np.pi*v))) * np.exp(-(((x-u)**2)/(2*v)))

    def predict(self, X):
        eps = np.finfo(float).eps
        n_samples, n_features = X.shape
        y_probs = np.zeros((n_samples, self.n_classes))
        for i, x_i in enumerate(X):
            for j, y_j in enumerate(self.classes):
                y_probs[i, j] = np.sum(np.log(self.gaussian(x_i, self.means[j], self.vars[j]))) + np.log(self.priors[j])
        return self.classes[np.argmax(y_probs, axis=1)]
        #return np.argmax(y_probs, axis=1)