
from sklearn.metrics import accuracy_score
from tensorflow      import keras

import numpy             as np
import matplotlib.pyplot as plt

from DenseLayer       import DenseLayer
from ReLU             import ReLU
from Sigmoid          import Sigmoid
from Tanh             import Tanh
from Softmax          import Softmax
from CrossEntropyLoss import CrossEntropyLoss

def main():

    n_classes  = 10
    alpha      = 0.10
    epochs     = 5_000
    batch_size = 10_000
    
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    X_train = X_train.reshape(60_000, 784)
    X_test  = X_test.reshape(10_000, 784)
    y_train = keras.utils.to_categorical(y_train, num_classes = n_classes)
    X_train = X_train / 255.
    X_test  = X_test  / 255.
    
    print(f"Training data shape:\t{X_train.shape}")
    print(f"Training labels shape:\t{y_train.shape}")
    print(f"Test data shape:\t{X_test.shape}")
    print(f"Test labels shape:\t{y_test.shape}")

    # create model
    nn = [
        DenseLayer(X_train.shape[1], 128, alpha=alpha),
        Sigmoid(),
        DenseLayer(128, 64, alpha=alpha),
        Sigmoid(),
        DenseLayer(64, 32, alpha=alpha),
        Sigmoid(),
        DenseLayer(32, 10, alpha=alpha)
    ]
    loss = CrossEntropyLoss()

    
    costs = []
    n_samples, n_features = X_train.shape
    # run for n epochs
    for e in range(epochs):
        # stochastic gradient descent
        for i in range(0, n_samples, batch_size):
            # grab x, y batches
            x_i = X_train[i:i+batch_size,:]
            y_i = y_train[i:i+batch_size,:]
            
            # forward
            out = x_i
            for layer in nn: out = layer.forward(out)
            
            # loss
            l  = loss.forward(out, y_i)
            dY = loss.backward()
            
            # backward
            for layer in reversed(nn): dY = layer.backward(dY)
            
            # update
            for layer in nn: layer.step()
            costs.append(l)
        print(e, costs[-1])
    
    # prediction
    out = X_test
    for layer in nn:
        out = layer.forward(out)
    y_pred = np.argmax(out, axis=1)
    print(y_pred.shape, y_test.shape)
    print(y_pred)
    print(y_test)
    print(f"accuracy: {accuracy_score(y_pred, y_test)}")
    
    # plot cost
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()