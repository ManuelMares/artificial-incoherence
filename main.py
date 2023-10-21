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
#%%
print("This is and example of an interactive cell in VSCode.")









# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
dfs = []
dfs.append(pd.read_csv("./DataSets/total_population.csv"))
dfs.append(pd.read_csv("./DataSets/Catholic_people.csv"))
dfs.append(pd.read_csv("./DataSets/foreign_migrant_population.csv"))
dfs.append(pd.read_csv("./DataSets/has_healthcare.csv"))
dfs.append(pd.read_csv("./DataSets/higher_education_population.csv"))
dfs.append(pd.read_csv("./DataSets/indigenous_status.csv"))
dfs.append(pd.read_csv("./DataSets/literacy_status.csv"))
dfs.append(pd.read_csv("./DataSets/working_age_population.csv"))
dfs.append(pd.read_csv("./DataSets/poverty_statistics.csv"))


#combines all the dfs into a single df
df = dfs[0]
for i in range(1,len(dfs)):
    df = pd.merge(df, dfs[i], on='National Urban System ID', how="left")


#delete duplicated columns
#%%
df = df.loc[:,~df.columns.duplicated()].copy()

# df
df.to_csv("./DataSets/main_dataset.csv")



# %%
# df = pd.read_csv("./DataSets/main_dataset.csv")
#checking if there are null values: no null values
# df.isna().sum()
# df.corr()


# %%
df.hist(figsize=(20,20), bins=5)

# getting relative values
# these are the columns that will be converted into relative values from the total population of the goegraphical area
dfs_columns = ["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]
population = df["Population"]
for col in dfs_columns:
    #divides the col in dfs_columns by the populaiton column
    df[col] = df[col].div(population, axis=0)



df.to_csv("./DataSets/normalized_data.csv")




# %%
df = pd.read_csv("./DataSets/normalized_data.csv")


dfs_columns = df[["Working Population", "Amount Catholic Population", "Foreign Migrant Population", "Has Healthcare", "Higher Education", "Amount of Indigenous Population", "Amount of Literate Population", "Poverty", "Population with at least 1 Social Lack", "Income below Welfare Line"]]
plt.figure(figsize=(20,20))
dfs_columns.boxplot()

# %%
Q1 = dfs_columns.quantile(0.25)
Q3 = dfs_columns.quantile(0.75)
IQR = Q3 - Q1
Lower_fence = Q1 - (1.5 * IQR)
Upper_fence = Q3 + (1.5 *IQR)

print(Lower_fence)
print(Upper_fence)



# %%

df.corr()
# %%
df.hist(figsize=(20,20), bins=5)

# %%
