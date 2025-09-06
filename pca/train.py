from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler, scale
from sklearn.datasets        import make_classification
from sklearn.datasets        import load_iris
from sklearn.metrics         import accuracy_score

from pca import PCA

import matplotlib.pyplot as plt

def train():
    n_samples     = 2_000
    n_features    = 6
    n_informative = 3
    n_classes     = 2
    test_size     = 0.2
    random_state  = 42
    
    X, y = make_classification(
        n_samples     = n_samples, 
        n_features    = n_features, 
        n_informative = n_informative, 
        n_classes     = n_classes,
        random_state  = random_state
    )
    
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, 
            y, 
            test_size    = test_size, 
            random_state = random_state
        )
    
    X, y = load_iris(return_X_y=True)
   
    pca = PCA(n_components=2)

    res = pca.fit(X)
    x1 = res[:,0] ; x2 = res[:,1]
    plt.scatter(x1, x2, c=y)
    plt.show()

if __name__ == '__main__':
    train()