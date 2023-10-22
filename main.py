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

#imports for metrics evaluation
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder

#imports for models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.svm import SVC








#%%
#==================================================================================
#global variables
#the main dataset, already normalized
DF = pd.read_csv("./DataSets/normalized_data.csv")
#A dataset with only the feature columns
_FEATURE_COLUMNS = DF[["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]]
#An array string containing the name of the feature columns
_FEATURE_COLUMNS_NAMES = ["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]










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




'''Plots scatterplots of data features - comparing one to another to visualize correlation.
@param dataset
@postcondition
    Scatter plots of features will be printed in the interactive window'''

def principle_component(dataset=_FEATURE_COLUMNS):

    pca = PCA(n_components=2)
    pca.fit(_FEATURE_COLUMNS)
    projected = pca.transform(_FEATURE_COLUMNS)
    projected = pd.DataFrame(projected, columns=['pc1', 'pc2'], index=range(1, 401 + 1))

    projected
    print(projected)

    plt.figure(figsize=(15,15))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

#%%
#==================================================================================
#code
# boxplot()
# histogram()
principle_component()



# %%

'''Data preparation for classication. '''
label = DF[['National Urban System ID']]
for each in range(len(label)):
    label = label.replace(to_replace=r'P.*', value='P', regex=True)
    label = label.replace(to_replace=r'C.*', value='C', regex=True)
    label = label.replace(to_replace=r'M.*', value='M', regex=True)

DF['National Urban System ID'] = label
print(DF['National Urban System ID'])
DF.to_csv('./DataSets/normalized_with_labels.csv')

# %%
