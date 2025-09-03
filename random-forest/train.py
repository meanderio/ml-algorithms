from RandomForest import RandomForest

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler, scale
from sklearn.datasets        import make_classification
from sklearn.metrics         import accuracy_score

def train():
    n_samples     = 2_000
    n_features    = 6
    n_informative = 3
    n_classes     = 2
    test_size     = 0.2
    random_state  = 42

    n_trees     = 10
    min_samples = 2
    max_depth   = 10
    
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
    
    #scaler  = MinMaxScaler().fit(X_train, y_train)
    #X_train = scaler.transform(X_train)
    #X_test  = scaler.transform(X_test)

    #X_train = scale(X_train)
    #X_test  = scale(X_test)
    
    clf = RandomForest(
        n_trees     = n_trees,
        min_samples = min_samples,
        max_depth   = max_depth
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print(y_pred)
    #print(y_test)
    print(f"accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}")

if __name__ == '__main__':
    train()