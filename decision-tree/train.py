from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from DecisionTreeClassifier import DecisionTreeClassifier as ACDT

from sklearn.datasets import make_classification
from sklearn.tree     import DecisionTreeClassifier as SKDT
from sklearn.metrics  import accuracy_score

def train():
    n_samples     = 1_000
    n_features    = 16
    n_informative = 12
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
            test_size=test_size, 
            random_state=random_state
        )
    
    scaler  = MinMaxScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    
    sk_dt = SKDT(max_depth=8)
    ac_dt = ACDT(max_depth=8)

    ac_dt.fit(X_train, y_train)
    y_pred = ac_dt.predict(X_test)
    print(f"accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}")

    sk_dt.fit(X_train, y_train)
    y_pred = sk_dt.predict(X_test)
    print(f"accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}")   
    return None

if __name__ == '__main__':
    train()