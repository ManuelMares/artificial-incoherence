"""
    To properly work in this project, don't forget to read the readme


    Important dates:
    October 22, midterm report
        • Data
            • What kind of data?
            • Dataset information such as number of instances, number of attributes,
            type of attributes
            • How do you process your data?
            • Which attributes you use and which one you don’t use? Why?
        • Data mining task
            • What task? Classification, Clustering, Anomaly detection,...
        • Progress
            • Which algorithms you have tried?
            • How are the preliminary results (e.g., accuracies, running time)?
        • Are there any challenges or difficulties that you are facing?
        • Schedule

"""



#==============================================================================
#======================BUILDING MAIN CSV=======================================
#==============================================================================
#THERE IS NO NEED TO RUN ANY OF THESE COMMENTED CODE
#IT IS LEFT HERE AS EVIDENCE AND REFERENCE TO OUR PREVIOUS WORK


#%%
#this section loads the different datasets
# dfs = []
# dfs.append(pd.read_csv("./DataSets/total_population.csv"))
# dfs.append(pd.read_csv("./DataSets/Catholic_people.csv"))
# dfs.append(pd.read_csv("./DataSets/foreign_migrant_population.csv"))
# dfs.append(pd.read_csv("./DataSets/has_healthcare.csv"))
# dfs.append(pd.read_csv("./DataSets/higher_education_population.csv"))
# dfs.append(pd.read_csv("./DataSets/indigenous_status.csv"))
# dfs.append(pd.read_csv("./DataSets/literacy_status.csv"))
# dfs.append(pd.read_csv("./DataSets/working_age_population.csv"))
# dfs.append(pd.read_csv("./DataSets/poverty_statistics.csv"))

#combines all the dfs into a single df
# df = dfs[0]
# for i in range(1,len(dfs)):
#     df = pd.merge(df, dfs[i], on='National Urban System ID', how="left")


#delete duplicated columns
# df = df.loc[:,~df.columns.duplicated()].copy()
#store main dataset
# df.to_csv("./DataSets/main_dataset.csv")


#checking if there are null values: no null values
# df = pd.read_csv("./DataSets/main_dataset.csv")
# df.isna().sum()
# df.corr()

#printing a histogram of values
# df.hist(figsize=(20,20), bins=5)

#Normalizing the values (relative to total population column)
# dfs_columns = ["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]
# population = df["Population"]
# for col in dfs_columns:
#     #divides the col in dfs_columns by the population column
#     df[col] = df[col].div(population, axis=0)
#Creating normalized csv
# df.to_csv("./DataSets/normalized_data.csv")


#cleaning the data from outliers.
#Outliers are those values above 1. Since the dataset was normalized (1 =100%)
#Outliers were replaced with zero-values
# for col in dfs_columns:
#     df[col].mask(df[col] >= 1, 0, inplace=True)

## %%

##Determines the 'National Urban System ID' as the classification label.
# label = df[['National Urban System ID']]
# for each in range(len(label)):
#     label = label.replace(to_replace=r'P.*', value='P', regex=True)
#     label = label.replace(to_replace=r'C.*', value='C', regex=True)
#     label = label.replace(to_replace=r'M.*', value='M', regex=True)

# df['National Urban System ID'] = label
# df.to_csv('./DataSets/normalized_with_labels.csv')

#splits dataset into training and testing sets
# def train_test(label, dataset=DF, test_size=0.2):
#     #splitting into train and test sets
#     train, test = train_test_split(dataset, test_size=test_size)

#     train.to_csv("./train.csv")
#     test.to_csv("./test.csv")




#==================================================================================
#===================EXPERIMENTING WITH FINAL DATASET===============================
#==================================================================================

# %%
#==================================================================================
#imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sb

#imports for metrics evaluation
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder

#imports for models
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.svm import SVC








#%%
#==================================================================================
#global variables
#the main dataset, already normalized
DF = pd.read_csv("./DataSets/normalized_with_labels.csv")
#A dataset with only the feature columns
_FEATURE_COLUMNS = DF[["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]]
#An array string containing the name of the feature columns
_FEATURE_COLUMNS_NAMES = ["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]
#training set
train = pd.read_csv("./DataSets/train.csv")
test = pd.read_csv("./DataSets/test.csv")









#%%
#==================================================================================
#functions
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

"""
Splits a dataset and returns X_train, y_train, X_test, y_test
@param label
    The feature to predict
@param dataset
    The dataset to use. Default value is DF
@param test_size
    The size of the test dataset. The default value is 0.2
@return  X_train
    A dataset with the features for training
@return  y_train
    A dataset with the target for training
@return  X_test
    A dataset with the features for testing
@return  y_test
    A dataset with the target for testing
"""
def split_labels(label, train, test):

    #splitting into labels and features
    X_train, y_train = train.loc[:, train.columns != label], train.loc[:,train.columns  == label]
    X_test, y_test = test.loc[:, test.columns != label], test.loc[:,test.columns  == label]

    return X_train, y_train, X_test, y_test

"""
Given a k range, this function trains a KNN model to identify the most option k.
@param k_range
    The array with the k values to test
@param X_train
    The train dataset of features
@param y_train
    The target dataset
@param fold
    The fold value for the knn model
@return
    The best k value
"""
def get_best_K(k_range, X_train, y_train, fold):
    best_accuracy = 0
    best_k = 0
    #print(X_train, y_train)
    for k in range(k_range[0], k_range[1]+1):
        #print("this is k ", k, " best k so far: ", best_k )
        knn = KNN(n_neighbors=k)
        #this function returns 5 values for each of the 5 folds. The mean gives the average value
        accuracy = cross_val_score(knn, X=X_train, y=y_train.values.ravel(), cv=fold ).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k



"""
Trains the best knn model and return the accuracy
@param dataset
    The dataset to use. Default value is DF
@param label
    The feature to predict
@return
    An integer indicating the accuracy
"""
def knn(label, dataset=DF):
    X_train, y_train, X_test, y_test = split_labels(label=label, train=train, test=test)
    k_test = (1,50)
    fold = 5
    best_k = get_best_K(k_test, X_train, y_train, fold)

    print(best_k)
    knn = KNN(n_neighbors=best_k)
    knn.fit(X_train, y_train.values.ravel())
    accuracy = knn.score(X_test, y_test)
    return accuracy

def lr(label, dataset=DF):
    X_train, y_train, X_test, y_test = split_labels(label=label, train=train, test=test)

    lc = LinearRegression()
    lc.fit(X=X_train, y=y_train)
    accuracy = lc.score(X_test, y_test)
    return accuracy
'''Plots Principal Component Analysis
@param dataset
@postcondition
    '''

def principle_component(dataset=_FEATURE_COLUMNS):

    pca = PCA(n_components=2)
    pca.fit(_FEATURE_COLUMNS)
    projected = pca.transform(_FEATURE_COLUMNS)
    projected = pd.DataFrame(projected, columns=['pc1', 'pc2'], index=range(1, 401 + 1))
    projected['Regions'] = DF[['National Urban System ID']]


    plt.figure(figsize=(15,15))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    targets = ['0', '1', '2']
    colors = ['b', 'r', 'g']
    for target, colors in zip(targets, colors):
        d = projected[projected['Regions'] == target]
        plt.scatter(d['pc1'], d['pc2'], c=colors, s=50)
    plt.legend(targets)


#%%
#==================================================================================
#code
# boxplot()
# histogram()
principle_component()


#%%
DF.corr()
heatmap()





# %%

#knn to predict higher education
df_higher_education = DF
df_higher_education = df_higher_education.loc[:, df_higher_education.columns != "National Urban System_x"]
#df_higher_education = df_higher_education.loc[:, df_higher_education.columns != "National Urban System ID"]

# accuracy = knn(label = "Higher Education", dataset=df_higher_education)
accuracy = knn(label = "National Urban System ID", dataset=df_higher_education)
print("Test set accuracy: ", accuracy)

# df_higher_education = df_higher_education.loc[:, df_higher_education.columns != "Income below Welfare Line"]
# # df_higher_education = df_higher_education.loc[:, df_higher_education.columns != "Population with at least 1 Social Lack"]
# accuracy = lr(label = "Poverty", dataset=df_higher_education)
# print("Test set accuracy: ", accuracy)

# %%
