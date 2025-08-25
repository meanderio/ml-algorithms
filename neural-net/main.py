from NeuralNet import NeuralNetwork

from sklearn.datasets        import make_classification
from sklearn.preprocessing   import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score

def main():
    n_samples    = 20_000
    n_features   = 24
    n_redundant  = 2
    n_classes    = 2
    random_state = 42
    n_iterations = 10_000
    
    X, y = make_classification(
        n_samples     = n_samples,
        n_features    = n_features,
        n_redundant   = n_redundant,
        n_informative = n_features - n_redundant,
        n_classes     = n_classes,
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
        8,
        8,
        1 # output layer, binary classifier [ 0, 1 ]
    ]

    ac_clf = NeuralNetwork(layers=layers, n_iters=n_iterations)
    ac_clf.fit(X_train, y_train)
    y_pred = ac_clf.predict(X_test).reshape(-1)
    print(f"accuracy: {accuracy_score(y_pred, y_test)}")

if __name__ == '__main__':
    main()