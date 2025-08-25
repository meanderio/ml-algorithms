from NeuralNet import NeuralNetwork

from sklearn.datasets        import make_classification
from sklearn.preprocessing   import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score

import matplotlib.pyplot as plt

def main():
    n_samples    = 12_000
    n_features   = 2
    n_redundant  = 0
    n_classes    = 2
    random_state = 42
    alpha        = 0.1
    n_iterations = 100
    
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

    layers = [
        X.shape[1], # input layer, use features
        3,
        1 # output layer, binary classifier [ 0, 1 ]
    ]

    ac_clf = NeuralNetwork(layers=layers, alpha=alpha, n_iters=n_iterations)
    ac_clf.fit(X_train, y_train)
    y_pred = ac_clf.predict(X_test).reshape(-1)
    print(f"accuracy: {accuracy_score(y_pred, y_test)}")
    plt.plot(ac_clf.costs)
    plt.show()

if __name__ == '__main__':
    main()