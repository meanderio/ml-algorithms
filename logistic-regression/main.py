from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from LogisticRegression   import LogisticRegression as ACLR
from sklearn.linear_model import LogisticRegression as SKLR

def main():
    print("logistic-regression")
    
    X, y = make_classification(
        n_samples = 2_000,
        n_features = 15,
        n_informative = 12,
        n_redundant = 3,
        n_classes = 2,
        weights = [0.5,0.5],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42
    )


    scaler         = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    m1 = ACLR()

    m1.fit(X_train_scaled, y_train)
    y_pred1  = m1.predict(X_test_scaled)
   
    print(f"accuracy: {accuracy_score(y_test, y_pred1)}")

    m2 = SKLR()
    m2.fit(X_train_scaled, y_train)
    y_pred2 = m2.predict(X_test_scaled)

    print(f"accuracy: {accuracy_score(y_test, y_pred2)}")



if __name__ == "__main__":
    main()
