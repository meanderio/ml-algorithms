
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier as SKKNC
from KNeighborsClassifier import KNeighborsClassifier as ACKNC

if __name__ == '__main__':

    n_samples = 1_000
    k_range   = 6
    #X, y = make_blobs(n_samples=n_samples, centers=4, n_features=2, random_state=42)
    X, y = make_classification(n_samples=n_samples, n_features=16, n_informative=12, n_classes=2)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    for k in range(1, k_range):
        ac_clf = ACKNC(k=k)
        sk_clf = SKKNC(n_neighbors=k)

        ac_clf.fit(X_train, y_train)
        y_pred1 = ac_clf.predict(X_test)
        print(f"{k} : {accuracy_score(y_test, y_pred1)}")

        sk_clf.fit(X_train, y_train)
        y_pred2 = sk_clf.predict(X_test)
        print(f"{k} : {accuracy_score(y_test, y_pred1)}")
        print()