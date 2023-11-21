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

#imports for metrics evaluation
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder

#imports for models
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
# from sklearn.svm import SVC

#%%
DF = pd.read_csv("./df_2015_normalized.csv")
_FEATURE_COLUMNS = DF[["indigenous", "Poverty", "higher_education", "healthcare","age_15_29", "age_30_44", "age_45_59", "age_60_75"]]

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
boxplot()

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

# %%
histogram()

# %%
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
heatmap()

# %%
DF.corr()

# %%
def spectral():
    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    silhouette_avg = []
    for k in cluster_range:
        sil = []
        for i in range(4):
            model = SpectralClustering(n_clusters=4, assign_labels='kmeans', n_init=5, random_state=42, affinity='nearest_neighbors')
            model.fit(DF)
            labels = model.labels_
            silhouette = silhouette_score(DF, labels, metric='euclidean')
            sil.append(silhouette)
        silhouette_avg.append(np.mean(sil))

    plt.figure(figsize=(20,20))
    plt.plot(cluster_range, silhouette_avg, 'bx-')
    plt.xlabel('K Values')
    plt.ylabel('Silhouette score')
    plt.show()

# %%
spectral()

# %%
def KMeansClustering():

    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    silhouette_avg = []

    #test to see the best k-value
    # for k in cluster_range:
    #     sil = []
    #     for i in range(1, 5):
    #         kmeans = KMeans(n_clusters=k, n_init=10, max_iter=5)
    #         kmeans.fit_transform(DF)

    #         silhouette = silhouette_score(DF, kmeans.labels_)
    #         sil.append(silhouette)

    #     silhouette_avg.append(np.mean(sil))

    # #plots the silhouette average values
    # plt.figure(figsize=(20,20))
    # plt.plot(cluster_range, silhouette_avg, 'bx-')
    # plt.xlabel('K Values')
    # plt.ylabel('Silhouette score')
    # plt.show()

    # print("The best K-value is 20.")

    #Fit the best silhouette average into a kmeans model
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=5)
    kmeans.fit_transform(DF)
    centroids = kmeans.cluster_centers_
    #print("These are the centroids: ", centroids)

    #Reduce the dimensions in order to plot the figure of kmeans model
    pca = PCA(n_components=2)
    pca.fit(centroids)
    projected = pca.transform(centroids)
    projected = pd.DataFrame(projected, columns=['pc1', 'pc2'], index=range(1, 6 + 1))
    projected['clusterID'] = ['1', '2', '3', '4', '5', '6']


    plt.figure(figsize=(15,15))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    targets = ['1', '2', '3', '4', '5', '6']
    for target in targets:
        d = projected[projected['clusterID'] == target]
        plt.scatter(d['pc1'], d['pc2'], s=50)
    plt.legend(targets, loc='lower right')
    plt.savefig('./KMeansClustering')

    X_train_pca = pca.fit_transform(DF)

    label = kmeans.fit_predict(DF)
    data = {
        'label': label,
        'pca1' : X_train_pca[:,0],
        'pca2' : X_train_pca[:,1],
    }
    clusters = pd.DataFrame(data)
    unique_categories = [x for x in range(len(DF))]
    colors = plt.cm.tab20.colors[:len(unique_categories)]

    for i, category in enumerate(unique_categories):
        plt.scatter(clusters[clusters['label'] == category]['pca1'],
                    clusters[clusters['label'] == category]['pca2'],
                    label=category,
                    color=colors[i],
                    alpha=0.7)
    plt.legend()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Scatter plot for the K-means model')
    plt.show()

    # df = copy.deepcopy(DF)
    # df['labels'] = label
    # tsne(df)
# %%
KMeansClustering()

# %%
def dbscan():
    db = DBSCAN(eps=0.3, min_samples=10)
    clusters = db.fit(DF)
    # silhouette = silhouette_score(DF, clusters.labels_)
    # print("Silhouette coefficient for DBSCAN: ", silhouette)

    labels = clusters.labels_
    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise = list(labels).count(-1)
    print(labels)
    # print("Estimated number of clusters: %d" % n_clusters)
    # print("Estimated number of noise points: %d" % n_noise)

    # y_pred = db.fit_predict(DF)
    # # plt.scatter(DF.iloc[:, 0], DF.iloc[:,1], c=y_pred, cmap='Paired')
    # # plt.title("DBSCAN")

    # pca = PCA(n_components=2)
    # X_train_pca = pca.fit_transform(DF)

    #labels = db.fit_predict(DF)
    # data = {
    #     'label': label,
    #     'pca1' : X_train_pca[:,0],
    #     'pca2' : X_train_pca[:,1],
    # }
    # clusters = pd.DataFrame(data)
    # unique_categories = [x for x in range(len(DF))]
    # colors = plt.cm.tab20.colors[:len(unique_categories)]

    # for i, category in enumerate(unique_categories):
    #     plt.scatter(clusters[clusters['label'] == category]['pca1'],
    #                 clusters[clusters['label'] == category]['pca2'],
    #                 label=category,
    #                 color=colors[i],
    #                 alpha=0.7)

    df = copy.deepcopy(DF)
    df['labels'] = labels
    tsne(df)

# %%
def hier():
    hier = AgglomerativeClustering(n_clusters=9)
    y_pred = hier.fit_predict(DF)
    # plt.scatter(DF.iloc[:,0], DF.iloc[:,1],c=y_pred, cmap='Paired')
    # plt.legend()
    # plt.title("Hierarchical")

    # pca = PCA(n_components=2)
    # X_train_pca = pca.fit_transform(DF)

    label = hier.fit_predict(DF)
    # data = {
    #     'label': label,
    #     'pca1' : X_train_pca[:,0],
    #     'pca2' : X_train_pca[:,1],
    # }
    # clusters = pd.DataFrame(data)
    # unique_categories = [x for x in range(len(DF))]
    # colors = plt.cm.tab20.colors[:len(unique_categories)]

    # for i, category in enumerate(unique_categories):
    #     plt.scatter(clusters[clusters['label'] == category]['pca1'],
    #                 clusters[clusters['label'] == category]['pca2'],
    #                 label=category,
    #                 color=colors[i],
    #                 alpha=0.7)

    df = copy.deepcopy(DF)
    df['labels'] = y_pred
    tsne(df)
# %%
hier()

# %%
from sklearn.manifold import TSNE
import plotly.express as px
'''
    @param: df is a deepcopy of the DF + the labels that are created when
            using cluster algorithms
    @postcondition: Plot of multidimensional data onto 2-D.
'''
def tsne(df):
    col = ["indigenous", "Poverty", "higher_education", "healthcare","age_15_29", "age_30_44", "age_45_59", "age_60_75"]
    ts_embed = TSNE(n_components=2).fit_transform(df)
    df["x_component"] = ts_embed[:,0]
    df["y_component"] = ts_embed[:,1]
    for each in col:
        fig = px.scatter(df, x="x_component", y="y_component",
                         hover_name="population", color=each, size_max=60)
        fig.update_layout(height=800)
        fig.show()

# %%
_FEATURE_COLUMNS = DF[["indigenous", "Poverty", "higher_education", "healthcare","age_15_29", "age_30_44", "age_45_59", "age_60_75"]]
col = DF[["population", "indigenous", "higher_education"]]
plt.figure(figsize=(20,20))
parallel_coordinates(DF, 'population')

# %%

dbscan()

# %%
