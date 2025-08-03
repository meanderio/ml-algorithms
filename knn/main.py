
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier as SKKNC
from KNeighborsClassifier import KNeighborsClassifier as ACKNC

import seaborn as sns
import numpy as np

def metrics(k, y, yp):
    print(
f"""\
{k} : \n\
{accuracy_score(y, yp)} \n\
{precision_score(y, yp)} \n\
{recall_score(y, yp)}\
"""
    )

if __name__ == '__main__':

    n_samples = 1_000
    k_values  = [i for i in range (1,31)]
    #X, y = make_blobs(n_samples=n_samples, centers=4, n_features=2, random_state=42)
    X, y = make_classification(n_samples=n_samples, n_features=16, n_informative=12, n_classes=2)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler  = MinMaxScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    scores1 = []
    scores2 = []
    for k in k_values:
        ac_clf = ACKNC(k=k)
        sk_clf = SKKNC(n_neighbors=k)

        ac_clf.fit(X_train, y_train)
        y_pred1 = ac_clf.predict(X_test)
        scores1.append(accuracy_score(y_test, y_pred1))
        #scores1.append(np.mean(cross_val_score(ac_clf, X, y, cv=5)))
        #metrics(k, y_test, y_pred1)
        
        
        sk_clf.fit(X_train, y_train)
        y_pred2 = sk_clf.predict(X_test)
        scores2.append(accuracy_score(y_test, y_pred2))
        #scores2.append(np.mean(cross_val_score(sk_clf, X, y, cv=5)))
        #metrics(k, y_test, y_pred2)

    sns.lineplot(x=k_values, y=scores1, label="mndrio")
    sns.lineplot(x=k_values, y=scores2, label="sklearn")
    plt.title('Comparison of KNN Implementations')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend() # Displays the labels defined in lineplot calls
    plt.show()
    plt.show()