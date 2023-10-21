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





# # %%
# #combines all the dfs into a single df
# df = dfs[0]
# for i in range(1,len(dfs)):
#     df = pd.merge(df, dfs[i], on='National Urban System ID', how="left")
# df

# # %%
# #delete duplicated columns
# df = df.loc[:,~df.columns.duplicated()].copy()
# df
# # %%
# df.to_csv("./DataSets/main_dataset.csv")













# %%
df = pd.read_csv("./DataSets/main_dataset.csv")
df