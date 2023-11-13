#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb




df_indigenous_2015          = pd.read_csv("./NewDataSets/SourceDataSets/indigenous_2015.csv")
df_literacy_2015            = pd.read_csv("./NewDataSets/SourceDataSets/literacy_2015.csv")
df_population_2015          = pd.read_csv("./NewDataSets/SourceDataSets/population_2015.csv")
df_poverty_2015             = pd.read_csv("./NewDataSets/SourceDataSets/poverty_2015.csv")
df_healthcare_2015          = pd.read_csv("./NewDataSets/SourceDataSets/healthcare_2015.csv")
df_higher_education_2015    = pd.read_csv("./NewDataSets/SourceDataSets/higher_education_2015.csv")
df_age_15_29_2015           = pd.read_csv("./NewDataSets/SourceDataSets/age_15_29_2015.csv")
df_age_30_44_2015           = pd.read_csv("./NewDataSets/SourceDataSets/age_30_44_2015.csv")
df_age_45_59_2015           = pd.read_csv("./NewDataSets/SourceDataSets/age_45_59_2015.csv")
df_age_60_75_2015           = pd.read_csv("./NewDataSets/SourceDataSets/age_60_75_2015.csv")


df1 =  df_indigenous_2015[df_indigenous_2015["Indigenous Self Abscribing"] == "Yes"]
df2 =  df_literacy_2015[df_literacy_2015["Literate"] == "Yes"]
df1 =  df1[                       ["Municipality ID", "Population"]]
df2 =  df2[                       ["Municipality ID", "Population"]]
df3 =  df_population_2015[        ["Municipality ID", "Population"]]
df4 =  df_poverty_2015[           ["Municipality ID", "Poverty"]]
df5 =  df_higher_education_2015[  ["Municipality ID", "Population"]]
df6 =  df_healthcare_2015[        ["Municipality ID", "Population"]]
df8  = df_age_15_29_2015[         ["Municipality ID", "Population"]]
df9  = df_age_30_44_2015[         ["Municipality ID", "Population"]]
df10 = df_age_45_59_2015[         ["Municipality ID", "Population"]]
df11 = df_age_60_75_2015[         ["Municipality ID", "Population"]]


df1 =   df1.rename(columns  = {"Population":  "indigenous"})
df2 =   df2.rename(columns  = {"Population":  "literacy"})
df3 =   df3.rename(columns  = {"Population":  "population"})
df4 =   df4.rename(columns  = {"Population":  "poverty"})
df5 =   df5.rename(columns  = {"Population":  "higher_education"})
df6 =   df6.rename(columns  = {"Population":  "healthcare"})
df8 =   df8.rename(columns  = {"Population":  "age_15_29"})
df9 =   df9.rename(columns  = {"Population":  "age_30_44"})
df10 =  df10.rename(columns = {"Population":  "age_45_59"})
df11 =  df11.rename(columns = {"Population":  "age_60_75"})


dff1  = pd.merge(df1,  df2, on="Municipality ID")
dff2  = pd.merge(df3,  df4, on="Municipality ID")
dff3  = pd.merge(df5,  df6, on="Municipality ID")
dff4  = pd.merge(df8,  df9, on="Municipality ID")
dff5  = pd.merge(df10,  df11, on="Municipality ID")

dfff1 = pd.merge(dff1,  dff2, on="Municipality ID")
dfff2 = pd.merge(dff3,  dff4, on="Municipality ID")

df_2015    = pd.merge(dfff1,  dfff2, on="Municipality ID")
df_2015    = pd.merge(df_2015,  dff5, on="Municipality ID")

df_2015

df_2015.to_csv("df_2015.csv")




df_indigenous_2020          = pd.read_csv("./NewDataSets/SourceDataSets/indigenous_2020.csv")
df_literacy_2020            = pd.read_csv("./NewDataSets/SourceDataSets/literacy_2020.csv")
df_population_2020          = pd.read_csv("./NewDataSets/SourceDataSets/population_2020.csv")
df_poverty_2020             = pd.read_csv("./NewDataSets/SourceDataSets/poverty_2020.csv")
df_catholic_2020            = pd.read_csv("./NewDataSets/SourceDataSets/catholic_2020.csv")
df_foreign_migrant_2020     = pd.read_csv("./NewDataSets/SourceDataSets/foreign_migrant_2020.csv")
df_healthcare_2020          = pd.read_csv("./NewDataSets/SourceDataSets/healthcare_2020.csv")
df_higher_education_2020    = pd.read_csv("./NewDataSets/SourceDataSets/higher_education_2020.csv")
df_age_15_29_2020           = pd.read_csv("./NewDataSets/SourceDataSets/age_15_29_2020.csv")
df_age_30_44_2020           = pd.read_csv("./NewDataSets/SourceDataSets/age_30_44_2020.csv")
df_age_45_59_2020           = pd.read_csv("./NewDataSets/SourceDataSets/age_45_59_2020.csv")
df_age_60_75_2020           = pd.read_csv("./NewDataSets/SourceDataSets/age_60_75_2020.csv")




df1  = df_indigenous_2020[df_indigenous_2020["Indigenous Self Abscribing"] == "Yes"]
df2  = df_literacy_2020[df_literacy_2020["Literate"] == "Yes"]

df1  = df1[                      ["Municipality ID", "Population"]]
df2  = df2[                      ["Municipality ID", "Population"]]
df3  = df_population_2020[       ["Municipality ID", "Population"]]
df4  = df_poverty_2020[          ["Municipality ID", "Poverty"]]
df5  = df_catholic_2020[         ["Municipality ID", "Population"]]
df6  = df_foreign_migrant_2020[  ["Municipality ID", "Population"]]
df7  = df_healthcare_2020[       ["Municipality ID", "Population"]]
df8  = df_higher_education_2020[ ["Municipality ID", "Population"]]
df9  = df_age_15_29_2020[        ["Municipality ID", "Population"]]
df10 = df_age_30_44_2020[        ["Municipality ID", "Population"]]
df11 = df_age_45_59_2020[        ["Municipality ID", "Population"]]
df12 = df_age_60_75_2020[        ["Municipality ID", "Population"]]



df1 =   df1.rename(columns  = {"Population":  "indigenous"})
df2 =   df2.rename(columns  = {"Population":  "literacy"})
df3 =   df3.rename(columns  = {"Population":  "population"})
df4 =   df4.rename(columns  = {"Population":  "poverty"})
df5 =   df5.rename(columns  = {"Population":  "catholic"})
df6 =   df6.rename(columns  = {"Population":  "foreign_migrant"})
df7 =   df7.rename(columns  = {"Population":  "healthcare"})
df8 =   df8.rename(columns  = {"Population":  "higher_education"})
df9 =   df9.rename(columns  = {"Population":  "age_15_29"})
df10 =  df10.rename(columns = {"Population":  "age_30_44"})
df11 =  df11.rename(columns = {"Population":  "age_45_59"})
df12 =  df12.rename(columns = {"Population":  "age_60_75"})


dff1  = pd.merge(df1,  df2,  on="Municipality ID")
dff2  = pd.merge(df3,  df4,  on="Municipality ID")
dff3  = pd.merge(df5,  df6,  on="Municipality ID")
dff4  = pd.merge(df7,  df8,  on="Municipality ID")
dff5  = pd.merge(df9,  df10, on="Municipality ID")
dff6  = pd.merge(df11, df12, on="Municipality ID")

dfff1 = pd.merge(dff1,  dff2, on="Municipality ID")
dfff2 = pd.merge(dff3,  dff4, on="Municipality ID")
dfff3 = pd.merge(dff5,  dff6, on="Municipality ID")

dffff1    = pd.merge(dfff1,  dfff2, on="Municipality ID")
df_2020       = pd.merge(dffff1,  dfff3, on="Municipality ID")

df_2020
df_2020.to_csv("df_2020.csv")






# %%
#Cleaning the data
import pandas as pd
df_2015 = pd.read_csv("./NewDataSets/SourceDataSets/df_2015.csv")
df_2020 = pd.read_csv("./NewDataSets/SourceDataSets/df_2020.csv")


df = pd.concat([df_2015, df_2020])
df.drop("Municipality ID", axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

colNames = ["population", "indigenous", "literacy", "Poverty", "higher_education", "healthcare", "age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic", "foreign_migrant"]
df = df[colNames]
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


#Normalizing the data
population = df["population"]
for col in colNames:
    #divides the col in dfs_columns by the population column
    df[col] = df[col].div(population, axis=0)
    df[col] = df[col].fillna(0)
    
df["population"] = population


#Encoding the population
label_population = pd.cut(x=df["population"], bins=[0,1000,100000,250000,1000000,5000000], labels = [1, 2, 3, 4, 5 ])
df["population"] = label_population


df.to_csv("df_normalized.csv")
df



# %%
