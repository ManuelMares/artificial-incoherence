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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import LabelEncoder

#imports for models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.svm import SVC












#%%
df_school_enrollment = pd.read_csv("./DataSets/academic_degree_enrollment.csv")
df_crime_records = pd.read_csv("./DataSets/crime_records.csv")
df_hospitals_location = pd.read_csv("./DataSets/hospitals_by_location.csv")
df_job_status = pd.read_csv("./DataSets/job_status.csv")
df_population_access = pd.read_csv("./DataSets/population_access.csv")
df_poverty = pd.read_csv("./DataSets/poverty.csv")
df_safety_perception = pd.read_csv("./DataSets/safety_perception.csv")
df_students_status = pd.read_csv("./DataSets/students_status.csv")
df_workforce_by_sector = pd.read_csv("./DataSets/workforce_by_sector.csv")




#%%
dfs = [df_school_enrollment, df_crime_records, df_hospitals_location, df_job_status, df_population_access, df_poverty, df_safety_perception, df_students_status, df_workforce_by_sector]

for df in dfs:
    # print(df.describe)
    print(df.dtypes, "\n\n")
# %%
