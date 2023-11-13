#%%
#--------------------------------------------------------------
#Imports
#--------------------------------------------------------------
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN




#%%
#--------------------------------------------------------------
#spliting the data
#--------------------------------------------------------------
df = pd.read_csv("./df_normalized.csv")
X = df.loc[:, df.columns != "population"]
y = df.loc[:, df.columns == "population"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#%%
#--------------------------------------------------------------
#Preparing the data for PCA
#--------------------------------------------------------------
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)






#%%
#--------------------------------------------------------------
#k-means
#--------------------------------------------------------------
k_ks            = [x for x in range(2,20)]
k_bestK         = 0
k_bestAverage   = 0
k_averages      = []
for k in k_ks:
    #training the model 5 times to get the average
    scores = []
    for time in range(0,5):
        kmeans = KMeans(n_clusters=k).fit(X_train)
        scores.append(silhouette_score(X_train, kmeans.labels_, metric='euclidean'))
    #storing the averages for future plotting
    average = np.average(scores)
    k_averages.append(average) 
    #selecting the best model
    if average > k_bestAverage:
        k_bestAverage = average
        k_bestK = k


# printing the answers
plt.plot(k_averages, scalex=k_ks, scaley=k_averages)
plt.xticks(k_ks)
plt.show()
print(f"The best average was {k_bestAverage} with a value of k={k_bestK}")


kmeans = KMeans(n_clusters=k_bestK).fit(X_train)
centroids = kmeans.cluster_centers_
print("K-means centroids:\n", centroids)


#Labeling each element on the training set
label = kmeans.fit_predict(X_train)
#creating a dataset with pca and labels
data = {
    'label': label,
    'pca1': X_train_pca[:,0],
    'pca2': X_train_pca[:,1]
}
clusters = pd.DataFrame(data)

# Generate unique colors for each category (18 categories)
unique_categories = [x for x in range(k_bestK)]
colors = plt.cm.tab20.colors[:len(unique_categories)]  # Using a colormap with 18 distinct colors

for i, category in enumerate(unique_categories):
    plt.scatter(clusters[clusters['label'] == category]['pca1'], 
                clusters[clusters['label'] == category]['pca2'],
                label=category,
                color=colors[i],
                alpha=0.7)

# Show legend with category labels
plt.legend()

# Set plot labels and title
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Scatter plot for the K-means model')

# Show the plot
plt.show()

























#%%
#--------------------------------------------------------------
#Gaussian Mixture
#--------------------------------------------------------------



ks = [x for x in range(2,21)]
bestK_gm = -10
bestAverage_gm = -10
averages_gm = []
for k in ks:
    #training the model 5 times to get the average
    scores = []
    for time in range(0,5):
        gm = GaussianMixture(n_components=k).fit(X_train)
        scores.append(silhouette_score(X_train, gm.predict(X_train)))
    
    #storing the averages for future plotting
    average_gm = np.average(scores)
    averages_gm.append(average_gm) 
    #selecting the best model
    if average_gm > bestAverage_gm:
        bestAverage_gm = average_gm
        bestK_gm = k


plt.plot(averages_gm, scalex=ks, scaley=averages_gm)
plt.xticks(ks)
plt.show()
print(f"The best average was {bestAverage_gm} with a value of k={bestK_gm}. However, a technically equally good value occurs at 19 clusters.")



#=======================================================================
#2b
gmm = GaussianMixture(n_components=bestK_gm).fit(X_train)
# centroids_gmm = gmm.means_
# print("Gaussian Mixture centroids:\n", centroids_gmm)


#2c

#Labeling each element on the training set
label_gmm = gmm.fit_predict(X_train)
#creating a dataset with pca and labels
data_gmm = {
    'label': label_gmm,
    'pca1': X_train_pca[:,0],
    'pca2': X_train_pca[:,1]
}
clusters_gmm = pd.DataFrame(data_gmm)

# Generate unique colors for each category (18 categories)
unique_categories_gmm = [x for x in range(bestK_gm)]
colors = plt.cm.tab20.colors[:len(unique_categories_gmm)]  # Using a colormap with 18 distinct colors

for i, category in enumerate(unique_categories_gmm):
    plt.scatter(clusters_gmm[clusters_gmm['label'] == category]['pca1'], 
                clusters_gmm[clusters_gmm['label'] == category]['pca2'],
                label=category,
                color=colors[i],
                alpha=0.7)

# Show legend with category labels
plt.legend()

# Set plot labels and title
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Scatter plot for the Gaussian Mixture model')

# Show the plot
plt.show()
print(f"The best average of the silhouette score is {bestAverage_gm}, so it makes sense that the clustering is not good. However, unlike in kmeans, the existence of only 2 groups seems to make a little bit more sense graphically")

























#%%

ks = [x for x in range(2,21)]
bestK_sc = 0
bestAverage_sc = 0
averages_sc = []
for k in ks:
    #training the model 5 times to get the average
    scores = []
    for time in range(1,4):
        sc=SpectralClustering(n_clusters=k,assign_labels='kmeans',
                            n_init=5,random_state=42*k+time,affinity='nearest_neighbors')
        sc.fit(X_train)
        scores.append(silhouette_score(X_train, sc.labels_, random_state=42*k, metric='euclidean'))
    #storing the averages for future plotting
    average_sc = np.average(scores)
    averages_sc.append(average_sc) 
    #selecting the best model
    if average_sc > bestAverage_sc:
        bestAverage_sc = average_sc
        bestK_sc = k
    

plt.plot(averages_sc, scalex=ks, scaley=averages_sc)
plt.xticks(ks)
plt.show()
print(f"The best average was {bestAverage_sc} with a value of k={bestK_sc}")



#==========================================================================
#2b.b
sc=SpectralClustering(n_clusters=bestK_sc,assign_labels='kmeans',
                            n_init=5,random_state=42*bestK_sc+time,affinity='nearest_neighbors').fit(X_train)
#Labeling each element on the training set
# label_sc = sc.fit_predict(X_train)
# #setting a variable for the centroids
# centroids_sc = []
# #the centroid is the average for the values in each cluster
# for cluster_label in np.unique(label_sc):
#     #obtaining the mean of the cluster for this label
#     label_centroid = np.mean(X[label_sc == cluster_label], axis=0)
#     #adding the centroid to the centroids list
#     centroids.append(label_centroid)
# print("Spectral Clustering centroids:\n", centroids_sc)



#2b.c
# Labeling each element on the training set
label_sc = sc.fit_predict(X_train)
#creating a dataset with pca and labels
data_sc = {
    'label': label_sc,
    'pca1': X_train_pca[:,0],
    'pca2': X_train_pca[:,1]
}
clusters_sc = pd.DataFrame(data_sc)

# Generate unique colors for each category (18 categories)
unique_categories_sc = [x for x in range(bestK_sc)]
colors = plt.cm.tab20.colors[:len(unique_categories_sc)]  # Using a colormap with 18 distinct colors

for i, category in enumerate(unique_categories_sc):
    plt.scatter(clusters_sc[clusters_sc['label'] == category]['pca1'], 
                clusters_sc[clusters_sc['label'] == category]['pca2'],
                label=category,
                color=colors[i],
                alpha=0.7)

# Show legend with category labels
plt.legend()

# Set plot labels and title
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Scatter plot')

# Show the plot
plt.show()
print(f"The best average of the silhouette score is {bestAverage_sc}, so it makes sense that the clustering is not good. However, unlike in previous graphs, there seems to be a clearer division of physical values in the graph.")



















# %%
#==========================================================================
#3 DBSCAN
#==========================================================================

ks = [x for x in range(2,21)]
bestK_dbscan = 0
bestAverage_dbscan = 0
averages_dbscan = []
for k in ks:
    #training the model 5 times to get the average
    scores = []
    for time in range(1,4):
        dbscan=DBSCAN(eps=3, min_samples=k).fit(X_train)
        scores.append(silhouette_score(X_train, dbscan.labels_))
    #storing the averages for future plotting
    average_dbscan = np.average(scores)
    averages_dbscan.append(average_dbscan) 
    #selecting the best model
    if average_dbscan > bestAverage_dbscan:
        bestAverage_dbscan = average_dbscan
        bestK_dbscan = k
    

plt.plot(averages_dbscan, scalex=ks, scaley=averages_dbscan)
plt.xticks(ks)
plt.show()
print(f"The best average was {bestAverage_dbscan} with a value of k={bestK_dbscan}")


































# %%
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import concurrent.futures
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10_000)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

if __name__ == '__main__':

    start = time.perf_counter()

    algo_name = [BernoulliNB, RandomForestClassifier, SVC, SGDClassifier]

    def train(algo_name) :
        model = algo_name().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(model, accuracy_score(y_test, y_pred))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(train, algo_name)

    end = time.perf_counter()
    print(f'Program runtime is {round((end - start) * 1000 , 2)} ms')
