from sklearn.datasets        import make_classification
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import scale
from SVMClassifier           import SVMClassifier as ACSVC
from sklearn.svm             import SVC as SKSVC



def main():
    X, y = make_classification(
        n_samples     = 2_000,
        n_features    = 15,
        n_informative = 12,
        n_redundant   = 3,
        n_classes     = 2,
        weights       = [0.5,0.5],
        random_state  = 42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size    = 0.2,
        random_state = 42
    )

    y_train[y_train == 0] = -1
    y_test[ y_test  == 0] = -1

    X_train = scale(X_train)
    X_test  = scale(X_test)

    m1 = ACSVC()

    m1.fit(X_train, y_train)
    y_pred1  = m1.predict(X_test)
    print(f"accuracy: {accuracy_score(y_test, y_pred1)}")

    m2 = SKSVC(kernel='linear')

    m2.fit(X_train, y_train)
    y_pred2 = m2.predict(X_test)
    print(f"accuracy: {accuracy_score(y_test, y_pred2)}")

if __name__ == '__main__':
    main()