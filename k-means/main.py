import numpy as np

from matplotlib              import pyplot as plt
from sklearn.datasets        import make_blobs, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics         import accuracy_score, precision_score, recall_score
from sklearn.preprocessing   import scale

from sklearn.cluster import KMeans as SKKM
from KMeans          import KMeans as ACKM

import plotly.express       as px
import seaborn              as sns
import plotly.graph_objects as go

def main():
    n_samples  = 2_000
    n_features = 6
    n_centers  = 3
    max_iter   = 2_000

    X, y, centers = make_blobs(
        n_samples      = n_samples,
        n_features     = n_features,
        centers        = n_centers,
        return_centers = True,
        random_state   = 42
    )

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size    = 0.2,
        random_state = 42
    )

    ac_mdl = ACKM(n_clusters = n_centers, max_iter = max_iter)
    sk_mdl = SKKM(n_clusters = n_centers, max_iter = max_iter)

    ac_mdl.fit(X_train, y_train)
    y_pred = ac_mdl.predict(X_test)
    print("accuracy:", np.sum(y_pred == y_test) / len(y_test))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_test[y_pred == 0, 0], y=X_test[y_pred == 0, 1],
        mode='markers',marker_color='#DB4CB2',name='Iris-setosa'
    ))

    fig.add_trace(go.Scatter(
        x=X_test[y_pred == 1, 0], y=X_test[y_pred == 1, 1],
        mode='markers',marker_color='#c9e9f6',name='Iris-versicolour'
    ))

    fig.add_trace(go.Scatter(
        x=X_test[y_pred == 2, 0], y=X_test[y_pred == 2, 1],
        mode='markers',marker_color='#7D3AC1',name='Iris-virginica'
    ))

    fig.add_trace(go.Scatter(
        x=ac_mdl.centers[:, 0], y=ac_mdl.centers[:,1],
        mode='markers',marker_color='#CAC9CD',marker_symbol=4,marker_size=13,name='Centroids'
    ))

    fig.update_layout(template='plotly_dark',width=1000, height=500,)
    fig.show(renderer='iframe')

    sk_mdl.fit(X_train, y_train)
    y_pred = sk_mdl.predict(X_test)
    print("accuracy:", np.sum(y_pred == y_test) / len(y_test))

if __name__ == '__main__':
    main()
