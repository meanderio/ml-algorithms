
from sklearn.datasets        import make_regression
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from LinearRegression        import LinearRegression as ACLR
from sklearn.linear_model    import LinearRegression as SKLR

if __name__ == '__main__':

    X, y = make_regression(
        n_samples=1_000, 
        n_features=1, 
        noise=5, 
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

    # version from scratch
    model = ACLR()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"mse: {mean_squared_error(y_test, y_pred)}")
    print(f"r2:  {r2_score(y_test, y_pred)}")

    m2 = SKLR()
    m2.fit(X_train_scaled, y_train)
    y2_pred = m2.predict(X_test_scaled)


    print(f"mse: {mean_squared_error(y_test, y2_pred)}")
    print(f"r2:  {r2_score(y_test, y2_pred)}")