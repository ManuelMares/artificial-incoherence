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




# %%

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

df1  = df_indigenous_2020[       ["Municipality ID", "Population"]]
df2  = df_literacy_2020[         ["Municipality ID", "Population"]]
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
print(len(df_2015))
print(len(df_2020))

df1 = df_indigenous_2015[df_indigenous_2015["Indigenous Self Abscribing"] == "Yes"]
df1 = df1[["Municipality", "Population"]]
df2 = df_literacy_2015[df_literacy_2015["Literate"] == "Yes"]
df2 = df2[["Municipality", "Population"]]
df3 = df_population_2015[["Municipality", "Population"]]
df4 = df_poverty_2015[["Municipality", "Poverty"]]
df5 = df_higher_education_2015[["Municipality", "Population"]]
df6 = df_healthcare_2015[["Municipality", "Population"]]
df7 = df_population_2015[["Municipality", "Population"]]
df8  = df_age_15_29_2015[["Municipality", "Population"]]
df9  = df_age_30_44_2015[["Municipality", "Population"]]
df10 = df_age_45_59_2015[["Municipality", "Population"]]
df11 = df_age_45_59_2015[["Municipality", "Population"]]


#%%
df1 =   df1.rename(columns={"Population":  "Indigenous_status"})
df2 =   df2.rename(columns={"Population":  "literacy"})
df3 =   df3.rename(columns={"Population":  "population_number"})
df4 =   df4.rename(columns={"Population":  "poverty_2015"})
df5 =   df5.rename(columns={"Population":  "higher_education"})
df6 =   df6.rename(columns={"Population":  "healthcare"})
df7 =   df7.rename(columns={"Population":  "population"})
df8 =   df8.rename(columns={"Population":  "Age_15_29"})
df9 =   df9.rename(columns={"Population":  "Age_30_44"})
df10 =  df10.rename(columns={"Population": "Age_45_59"})
df11 =  df11.rename(columns={"Population": "Age_45_59"})



#%%
df = pd.merge(df1, df2, on="Municipality")
df

df = pd.merge(df,  df3, on="Municipality")


dd = df[df["Municipality"] == "Aguascalientes"]
print(len(dd))
print(len(df))
print(len(df3))
print(len(df["Municipality"].unique()))

df.to_csv("df")


#%%
df = pd.merge(df,  df4, on="Municipality")
df
#%%
df = pd.merge(df,  df5, on="Municipality")
df
#%%
df = pd.merge(df,  df6, on="Municipality")
df
#%%
df = pd.merge(df,  df7, on="Municipality")
df

#%%
df = pd.merge(df,  df8, on="Municipality")
df
#%%
df = pd.merge(df,  df9, on="Municipality")
df
#%%
df = pd.merge(df,  df10, on="Municipality")
df
#%%
df = pd.merge(df,  df11, on="Municipality")
df




#%%





df2 = df_indigenous_2020[df_indigenous_2020["Indigenous Self Abscribing"] == "Yes"]
df2 = df1[["Municipality", "Population"]]
df2

df_final = df2[["Municipality", "Population"]] + df1[["Municipality", "Population"]]
df_final


# %%
import pandas as pd
df1 = pd.read_csv("./df_2015")
df2 = pd.read_csv("./df_2020")


df = pd.concat([df1, df2])
df.drop("Municipality ID", axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

colNames = ["population", "indigenous", "literacy", "Poverty", "higher_education", "healthcare", "age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic",	"foreign_migrant"]
df = df[colNames]

print(len(df1))
print(len(df2))
# df.to_csv("")

# %%
