import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import seaborn as sns
import pandas as pd
from umap import UMAP
import umap
import seaborn as sns
from sklearn.svm import LinearSVC
from tqdm.notebook import tqdm

import plotly.express as px

def transformed_df(df):
    scaler = StandardScaler()
    X_cols = df.drop(columns=["label"]).columns
    scaled_df = df.copy()
    scaled_df[X_cols] = scaler.fit_transform(df[X_cols].values)
    return scaled_df


def draw_umap(df,
              n_neighbors=15,
              min_dist=0.0,
              n_components=3,
              metric='euclidean'):
    data= df.drop(columns=["label"]).values
    labels = df["label"].values
    title = "{} UMAP: {} kNN, min_dist={}, n={}".format(metric.upper(), n_neighbors, min_dist, n_components)
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=labels)
    if n_components == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=u[:,0], y=u[:,1], hue=labels, ax=ax, s=5)
    if n_components == 3:
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(u[:,0], u[:,1], u[:,2], c=labels, s=5)
        foo = pd.DataFrame({"x": u[:,0], "y": u[:,1], "z": u[:,2], "label": labels})
        fig = px.scatter_3d(foo, x='x', y='y', z='z',
                            marker=dict(color="label",
                            showscale=False))
        fig.update_traces(marker_size=1)
        fig.show()
    
    