from NaiveBayes              import NaiveBayes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import scale
from sklearn.datasets        import make_classification
from sklearn.metrics         import accuracy_score

def train():

    # input parameters
    n_samples     = 1_000
    n_features    = 10
    n_informative = 20
    n_classes     = 2
    test_size     = 0.2
    random_state  = 123
    
    # create fake data
    X, y = make_classification(
        n_samples     = n_samples, 
        n_features    = n_features, 
        #n_informative = n_informative, 
        n_classes     = n_classes,
        random_state  = random_state
    )
    
    # split data into train, test
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, 
            y, 
            test_size    = test_size, 
            random_state = random_state
        )
  
    # scale the data
    #X_train = scale(X_train)
    #X_test  = scale(X_test)
    
    # train the classifier
    clf = NaiveBayes()
    clf.fit(X_train, y_train)

    # predict on test data
    y_pred = clf.predict(X_test)
    print(f"accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}")

if __name__ == '__main__':
    train()