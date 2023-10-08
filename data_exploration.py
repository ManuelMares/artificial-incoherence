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
import seaborn as sns
plt.style.use('ggplot')
# pd.set_option('max_columns', 200)



#imports for metrics evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import LabelEncoder

#imports for models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.svm import SVC












#%%
#import data set
df_school_enrollment = pd.read_csv("./DataSets/academic_degree_enrollment.csv")
df_job_status = pd.read_csv("./DataSets/job_status.csv")
df_population_access = pd.read_csv("./DataSets/population_access.csv")
df_poverty = pd.read_csv("./DataSets/poverty.csv")
df_safety_perception = pd.read_csv("./DataSets/safety_perception.csv")
df_students_status = pd.read_csv("./DataSets/students_status.csv")



#%%
dfs = [df_school_enrollment, df_job_status, df_population_access, df_poverty, df_safety_perception, df_students_status]

for df in dfs:
    # print(df.describe)
    # print(df.dtypes)
    print(df.shape)
    # print(df.columns, "\n\n")
    # df.hist(bins=60, figsize=(20,10))
    print(len(df["National Urban System"].unique()))



# %%
for df in dfs:
    print(df.shape)
    print(df.dtypes)










# %%
#exploring df_population_access
df_population_access.describe()
df_population_access.head()
df_population_access.columns

# %%
