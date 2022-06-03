import plotly.express as px
import pandas as pd
import numpy as np
import umap
import streamlit as st
from utils import transformed_df, draw_umap
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
    st.subheader('Metric for mapping')
    umap__metric = st.sidebar.selectbox('This parameter controls how distance is computed in the ambient space of the input data:',
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

    st.subheader('Number of Neighbors for the Graph')
    umap__n_neighbors = st.sidebar.slider(
        label='This parameter controls how UMAP balances local versus global structure in the embeddings:',
        min_value=2,
        max_value=200,
        value=5,
        step=1)
    st.subheader('Threshold Distance')
    umap__min_dist = st.sidebar.slider('The min_dist parameter controls how tightly UMAP is allowed to pack points together:',
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

st.write(fig)