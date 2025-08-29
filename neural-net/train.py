
from sklearn.metrics import accuracy_score
from tensorflow      import keras

import matplotlib.pyplot as plt

from DenseLayer import DenseLayer
from ReLU       import ReLU

def main():

    n_classes  = 10
    alpha      = 0.1
    epochs     = 100
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
    
    layer_1 = DenseLayer(X_train.shape[1], 32)
    layer_1.forward(X_train)
    activ_1 = ReLU()
    activ_1.forward(layer_1.output)
    print(activ_1.output)

if __name__ == '__main__':
    main()