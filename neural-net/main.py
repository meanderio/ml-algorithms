from NeuralNet import NeuralNetwork

from sklearn.datasets        import make_classification
from sklearn.preprocessing   import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score

import tensorflow as tf
from   tensorflow import keras

import matplotlib.pyplot as plt

def main():
    n_samples    = 20_000
    n_features   = 12
    n_redundant  = 0
    n_classes    = 2
    random_state = 42
    alpha        = 0.1
    n_iterations = 5_000
    
    """
    X, y = make_classification(
        n_samples     = n_samples,
        n_features    = n_features,
        n_redundant   = n_redundant,
        n_informative = n_features - n_redundant,
        n_classes     = n_classes,
        weights       = [0.5, 0.5],
        random_state  = random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size    = 0.2,
        random_state = random_state
    )

    X_train = scale(X_train)
    X_test  = scale(X_test)
    """
    #"""
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60_000, 784)
    X_test  = X_test.reshape(10_000, 784)
    print(X_train.shape, y_train.shape)

    X_train = X_train / 255.
    X_test  = X_test  / 255.

    #X_train = scale(X_train)
    #X_test  = scale(X_test)
    #"""
    
    layers = [
        X_train.shape[1], # input layer
        8,
        4,
        10 # output layer
    ]

    ac_clf = NeuralNetwork(layers=layers, alpha=alpha, n_iters=n_iterations)
    ac_clf.fit(X_train, y_train)
    y_pred = ac_clf.predict(X_test).reshape(-1)
    print(f"accuracy: {accuracy_score(y_pred, y_test)}")
    plt.plot(ac_clf.costs)
    plt.show()

if __name__ == '__main__':
    main()