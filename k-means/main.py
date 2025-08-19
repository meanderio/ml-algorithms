import numpy   as np
import seaborn as sns

from matplotlib              import pyplot as plt
from sklearn.datasets        import make_blobs, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics         import accuracy_score, precision_score, recall_score
from sklearn.preprocessing   import scale

from sklearn.cluster import KMeans as SKKM
from KMeans          import KMeans as ACKM

def plot_step(X, y, centers):
        sns.scatterplot(
            x = [x[0] for x in X],
            y = [x[1] for x in X],
            hue     = y,
            palette = "deep",
            legend  = None
        )
        sns.scatterplot(
            x = [x[0] for x in centers],
            y = [x[1] for x in centers],
            color=".1", marker="+",
            palette = "deep",
            legend  = None
        )

        plt.title("Predictions")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

def main():
    n_samples   = 1_500
    n_features  = 4
    n_centers   = 4
    max_iter    = 2_000
    cluster_std = 1.2

    X, y, centers = make_blobs(
        n_samples      = n_samples,
        n_features     = n_features,
        centers        = n_centers,
        return_centers = True,
        random_state   = 42,
        cluster_std    = cluster_std
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size    = 0.2,
        random_state = 42
    )

    ac_mdl = ACKM(n_clusters = n_centers, max_iter = max_iter, plot=True)
    sk_mdl = SKKM(n_clusters = n_centers, max_iter = max_iter)

    ac_mdl.fit(X_train)
    y_pred, centers = ac_mdl.predict(X_test)
    plot_step(X_test, y_pred, centers)
    

    sk_mdl.fit(X_train, y_train)
    y_pred = sk_mdl.predict(X_test)
    plot_step(X_test, y_pred, sk_mdl.cluster_centers_)

if __name__ == '__main__':
    main()
