import plotly.express as px
import pandas as pd
import numpy as np
import umap
import streamlit as st
#from utils import transformed_df, draw_umap

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

def transformed_df(df):
    scaler = StandardScaler()
    X_cols = df.drop(columns=["label"]).columns
    print(X_cols[:10])
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
        plot = sns.scatterplot(x=u[:,0], y=u[:,1], hue=labels, ax=ax, s=5)
        return plot
    if n_components == 3:
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(u[:,0], u[:,1], u[:,2], c=labels, s=5)
        foo = pd.DataFrame({"x": u[:,0], "y": u[:,1], "z": u[:,2], "label": labels})
        fig = px.scatter_3d(foo, x='x', y='y', z='z',color=labels,
                             labels=labels, symbol=labels,)
        fig.update_coloraxes(showscale=False)
        fig.update_traces(marker_size=1)
        return fig
    
    


def call_streamlit():

    TITLE = 'LACE-UMAP: JET CLASSIFICATION'

    st.set_page_config(
        page_title=TITLE,
        page_icon='👾',
        layout='centered'
    )

    st.title(TITLE)

    st.markdown('LACE Embedding visualization')

    return st


@st.cache(persist=True)
def load_data(sample_size=10_000, seed=0):
    df = pd.read_csv('./data/aced_jets.csv')#.iloc[12_000:].reset_index(drop=True)
    tdf = transformed_df(df)
    return tdf



def get_parameters(st):
    # st.subheader('Choose distance metric')
    umap__metric = st.sidebar.selectbox('Metric for mapping: This parameter controls how distance is computed in the ambient space of the input data:',
                                        ['euclidean',
                                         'manhattan',
                                         'chebyshev',
                                         'minkowski',
                                         'canberra',
                                         'braycurtis',
                                         # 'haversine',
                                         'mahalanobis',
                                         'wminkowski',
                                         # 'seuclidean'
                                         ])

    umap__n_neighbors = st.sidebar.slider(
        label='Number of Neighbors for the Graph: This parameter controls how UMAP balances local versus global structure in the embeddings:',
        min_value=2,
        max_value=200,
        value=5,
        step=1)
    umap__min_dist = st.sidebar.slider('Threshold Distance: The min_dist parameter controls how tightly UMAP is allowed to pack points together:',
                                       min_value=.1,
                                       max_value=1.0,
                                       value=.5,
                                       step=.1)

    if umap__metric == None:
        umap__metric = 'euclidean'

    return umap__metric, umap__n_neighbors, umap__min_dist




st = call_streamlit()
df = load_data()
umap__metric, umap__n_neighbors, umap__min_dist = get_parameters(st)

fig = draw_umap(
    df,
    n_neighbors=umap__n_neighbors,
    min_dist=umap__min_dist,
    metric=umap__metric)
if fig != None:
    st.plotly_chart(fig)

#st.write(fig)