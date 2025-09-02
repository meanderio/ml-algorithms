import numpy as np

class DenseLayer:
    def __init__(self, m, n, alpha=0.01):
        self.weights = np.random.randn(m, n)
        self.biases  = np.zeros((1, n))
        self.alpha   = alpha
        self.dw = None
        self.db = None
    
    def forward(self, X):
        self.inputs = X
        self.output = X @ self.weights + self.biases
        return self.output

    def backward(self, dY):
        #print(dY.shape, self.weights.T.shape)
        N = self.inputs.shape[0]
        self.dw = (self.inputs.T @ dY) #/ N
        self.db = np.sum(dY, axis=0, keepdims=True) #/ N
        return dY @ self.weights.T
    
    def step(self,):
        self.weights -= self.alpha * self.dw
        self.biases  -= self.alpha * self.db