#see the job status of women vs men
#see the salaries of women vs men
#see age range of works with their salaries
# see schooing years with salaries
#see schooling years with work sector
# plot a histogram of outliers
# try to standardize information somehow

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

df = pd.read_csv("./data/Academic Degree Population.csv")
print(df.head)

#%%
highEdDf = df[df["Academic Degree ID"] > 5]
highEdDf = highEdDf[highEdDf["Academic Degree ID"] < 99]


aggregate_functions = {'National Urban System': 'first', 'Population': 'sum'}
aggregated_academic_population = highEdDf.groupby(df['National Urban System ID']).aggregate(aggregate_functions)
aggregated_academic_population = aggregated_academic_population.rename(columns={'Population': "Higher Education"})
aggregated_academic_population.to_csv('./DataSets/higher_education_population.csv')

# %%
df = pd.read_csv("./data/Age Range Population.csv")

df = df[df["Age Range ID"] > 3]
df = df[df["Age Range ID"] < 18]

aggregate_functions = {'National Urban System': 'first', 'Population': 'sum'}
aggregated_working_age = df.groupby(df['National Urban System ID']).aggregate(aggregate_functions)
aggregated_working_age = aggregated_working_age.rename(columns={'Population' : "Working Population"})
aggregated_working_age.to_csv('./DataSets/working_age_population.csv')

# %%
df = pd.read_csv("./data/Foreign Migrant Population.csv")
df2 = pd.read_csv("./DataSets/Total Population.csv")

df = df[df["Foreign Migrant ID"] == 0]
aggregate_functions = {'National Urban System': 'first', 'Population': 'first'}
df = df.groupby(df['National Urban System ID']).aggregate(aggregate_functions)

for i in range(len(df2["Population"])):
    fm = df2["Population"].loc[df2.index[i]] - df["Population"].loc[df.index[i]]
    df["Population"].loc[df.index[i]] = fm

df = df.rename(columns={'Population' : "Foreign Migrant Population" })
df.to_csv('./DataSets/foreign_migrant_population.csv')


# %%
df = pd.read_csv("./data/Healthcare Population.csv")
df = df[df["Health Care ID"] < 8]

aggregate_functions = {'National Urban System': 'first', 'Population': 'sum'}
aggregated_healthcare = df.groupby(df['National Urban System ID']).aggregate(aggregate_functions)
aggregated_healthcare = aggregated_healthcare.rename(columns={'Population': "Has Healthcare"})
aggregated_healthcare.to_csv('./DataSets/has_healthcare.csv')

# %%
