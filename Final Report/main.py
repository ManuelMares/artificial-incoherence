#%%
#imports
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sb
from sklearn.metrics import silhouette_score
from pandas.plotting import parallel_coordinates
import plotly.express as px

#imports for models
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.manifold import TSNE

#%%
DF = pd.read_csv("./df_normalized.csv")
_FEATURE_COLUMNS = DF[["indigenous", "Poverty", "higher_education", "healthcare","age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic", "foreign_migrant"]]


#%%

def normalize_dataset(df):
    """
        This method:
            drops the Municipality column
            encodes the population column
            normalizes all the columns by obtaining a relative value to the total population
            gets rid of outliers
        @param df
            The dataset to clean
        @return
            A normalized dataset
    """
    df.drop("Municipality ID", axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    colNames = ["population", "indigenous", "literacy", "Poverty", "higher_education", "healthcare", "age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic", "foreign_migrant"]
    df = df[colNames]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


    #Normalizing the data
    #Changing NaN for 0
    #Deleting outliers
    cols_to_normalize = ["indigenous", "literacy", "Poverty", "higher_education", "healthcare", "age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic", "foreign_migrant"]
    population = df["population"]
    for col in cols_to_normalize:
        #divides the col in dfs_columns by the population column
        df[col] = df[col].div(population, axis=0)
        df[col] = df[col].fillna(0)
        df[col].mask(df[col] >= 1, 0, inplace=True)


    #Encoding the population
    label_population = pd.cut(x=df["population"], bins=[0,1000,100000,250000,1000000,5000000], labels = [1, 2, 3, 4, 5 ])
    df["population"] = label_population

    return df



#%%
"""
Plots a box-plot figure and indicates the fence values for the outliers
@param columns (optional)
    A dataset consisting of only the features to plot
    If no parameter is given, the default value is _FEATURE_COLUMNS
@posconditions
    A box-plot figure is printed in the interactive window
"""
def boxplot(columns=_FEATURE_COLUMNS):
    #plotting figure
    plt.figure(figsize=(20,20))
    columns.boxplot()

    #obtaining limits for the outliers
    Q1 = columns.quantile(0.25)
    Q3 = columns.quantile(0.75)
    IQR = Q3 - Q1
    Lower_fence = Q1 - (1.5 * IQR)
    Upper_fence = Q3 + (1.5 *IQR)

    #printing the outliers
    print(Lower_fence)
    print(Upper_fence)

# %%
# %%
"""
Plots a histogram figure and indicates the fence values for the outliers
@param dataset (optional)
    The version of the data set to print
    If no parameter is given, the default value is the normalized version DF

@posconditions
    A histogram figure is printed in the interactive window
"""
def histogram(dataset=DF):
    dataset.hist(figsize=(20,20), bins=5)

"""
Plots a heatmap figure indicates the correlation between features
@param dataset (optional)
    The version of the data set to print
    If no parameter is given, the default value is the normalized version DF

@posconditions
    A heatmap figure is printed in the interactive window
"""
def heatmap(dataset=DF):
    # plotting correlation heatmap
    dataplot = sb.heatmap(dataset.corr(), cmap="YlGnBu", annot=True)
    # displaying heatmap
    plt.show()

# %%
    """
    Uses spectral clustering to visualize data
    @param df
        A dataset to analyze
        If no dataset is provided, DF is the default value
    """
def spectral(df = DF):
    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    silhouette_avg = []
    for k in cluster_range:
        sil = []
        for i in range(4):
            model = SpectralClustering(n_clusters=4, assign_labels='kmeans', n_init=5, random_state=42, affinity='nearest_neighbors')
            model.fit(df)
            labels = model.labels_
            silhouette = silhouette_score(df, labels, metric='euclidean')
            sil.append(silhouette)
        silhouette_avg.append(np.mean(sil))

    plt.figure(figsize=(20,20))
    plt.plot(cluster_range, silhouette_avg, 'bx-')
    plt.xlabel('K Values')
    plt.ylabel('Silhouette score')
    plt.show()

#%%
def KMeansClustering(df = DF):
    """
    KMeans clusters the dataset
    @param df
        a data frame
    """
     #Fit the best silhouette average into a kmeans model
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10, max_iter=5)
    kmeans.fit_transform(df)
    centroids = kmeans.cluster_centers_
    label = kmeans.fit_predict(df)

    df = copy.deepcopy(df)
    df['labels'] = label
    tsne(df)

# %%
def dbscan( df = DF):
    """
    Clusters the dataset
    @param df
        a data frame
    """
    db = DBSCAN(eps=0.3, min_samples=10)
    clusters = db.fit(df)
    # silhouette = silhouette_score(df, clusters.labels_)
    # print("Silhouette coefficient for DBSCAN: ", silhouette)

    labels = clusters.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    df = copy.deepcopy(df)
    df['labels'] = labels
    tsne(df)

# %%
def hier(df = DF):
    """
    Clusters the dataset
    @param df
        a data frame
    """
    hier = AgglomerativeClustering(n_clusters=9)
    label = hier.fit_predict(df)

    df = copy.deepcopy(df)
    df['labels'] = label
    tsne(df)

#%%
'''
    @param: df is a deepcopy of the DF + the labels that are created when
            using cluster algorithms
    @postcondition: Plot of multidimensional data onto 2-D.
'''
def tsne(df):
    ts_embed = TSNE(n_components=2).fit_transform(df)
    df["x_component"] = ts_embed[:,0]
    df["y_component"] = ts_embed[:,1]
    fig = px.scatter(df, x="x_component", y="y_component",
                     hover_name="population", color="indigenous", size_max=60)
    fig.update_layout(height=800)
    fig.show()

# %%
def para_coordinates(df = DF):
    _FEATURE_COLUMNS = df[["indigenous", "Poverty", "higher_education", "healthcare","age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic", "foreign_migrant"]]
    col = df[["population", "indigenous", "higher_education"]]
    plt.figure(figsize=(20,20))
    parallel_coordinates(df, 'population')
