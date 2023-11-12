#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb




df_indigenous_2015          = pd.read_csv("./NewDataSets/SourceDataSets/indigenous_2015.csv")
df_literacy_2015            = pd.read_csv("./NewDataSets/SourceDataSets/literacy_2015.csv")
df_population_2015          = pd.read_csv("./NewDataSets/SourceDataSets/population_2015.csv")
df_poverty_2015             = pd.read_csv("./NewDataSets/SourceDataSets/poverty_2015.csv")
df_work_situation_2015      = pd.read_csv("./NewDataSets/SourceDataSets/work_situation_2015.csv")
df_healthcare_2015          = pd.read_csv("./NewDataSets/SourceDataSets/healthcare_2015.csv")
df_higher_education_2015    = pd.read_csv("./NewDataSets/SourceDataSets/higher_education_2015.csv")


df_indigenous_2020             = pd.read_csv("./NewDataSets/SourceDataSets/indigenous_2020.csv")
df_literacy_2020            = pd.read_csv("./NewDataSets/SourceDataSets/literacy_2020.csv")
df_population_2020          = pd.read_csv("./NewDataSets/SourceDataSets/population_2020.csv")
df_poverty_2020             = pd.read_csv("./NewDataSets/SourceDataSets/poverty_2020.csv")
df_work_situation_2020      = pd.read_csv("./NewDataSets/SourceDataSets/work_situation_2020.csv")
df_catholic_2020            = pd.read_csv("./NewDataSets/SourceDataSets/catholic_2020.csv")
df_foreign_migrant_2020     = pd.read_csv("./NewDataSets/SourceDataSets/foreign_migrant_2020.csv")
df_healthcare_2020          = pd.read_csv("./NewDataSets/SourceDataSets/healthcare_2020.csv")
df_higher_education_2020    = pd.read_csv("./NewDataSets/SourceDataSets/higher_education_2020.csv")








# %%
# DATA SET FOR 2015

df1 = df_indigenous_2015[df_indigenous_2015["Indigenous Self Abscribing"] == "Yes"]
df1 = df1[["Municipality", "Population"]]
df2 = df_literacy_2015[["Municipality", "Population"]]
df3 = df_population_2015[["Municipality", "Population"]]
df4 = df_poverty_2015[["Municipality", "Poverty"]]
df5 = df_higher_education_2015[["Municipality", "Population"]]
df6 = df_healthcare_2015[["Municipality", "Population"]]
df7 = df_population_2015[["Municipality", "Population"]]
# df5 = df_work_situation_2015[["Municipality", "Poverty"]]



#%%





df2 = df_indigenous_2020[df_indigenous_2020["Indigenous Self Abscribing"] == "Yes"]
df2 = df1[["Municipality", "Population"]]
df2

df_final = df2[["Municipality", "Population"]] + df1[["Municipality", "Population"]]
df_final
# %%
