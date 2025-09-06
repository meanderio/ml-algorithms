import matplotlib.pyplot as plt
from   sklearn.datasets  import make_classification
from   pca               import PCA

def train():
    n_samples     = 200
    n_features    = 6
    n_informative = 4
    n_classes     = 2
    random_state  = 42
    
    X, y = make_classification(
        n_samples     = n_samples, 
        n_features    = n_features, 
        n_informative = n_informative, 
        n_classes     = n_classes,
        weights       = [0.5, 0.5],
        random_state  = random_state
    )
    
    # run pca
    pca = PCA(n_components=2)
    res = pca.fit(X)
    
    # plot results
    x1 = res[:,0] ; x2 = res[:,1]
    plt.scatter(x1, x2, c=y)
    plt.show()

if __name__ == '__main__':
    train()