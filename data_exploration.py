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











#%%
#exploring religion_status
df = pd.read_csv("./SourceDataSets/religion_status.csv")
Catholic_people = df[df["Religion"] == "Catholic"]
Catholic_people
aggregation_functions = {'National Urban System': 'first', "Population":"first"}
Catholic_people = df.groupby(Catholic_people['National Urban System ID']).aggregate(aggregation_functions)
Catholic_people = Catholic_people.rename(columns={'Population':"Amount Catholic Population"})
Catholic_people
Catholic_people.to_csv("./DataSets/Catholic_people.csv")






# %%
#exploring literacy_status
df = pd.read_csv("./SourceDataSets/literacy_status.csv")
literacy_status = df[df["Literate"] == "Yes"]
aggregation_functions = {'National Urban System': 'first', "Population":"first"}
literacy_status = df.groupby(literacy_status['National Urban System ID']).aggregate(aggregation_functions)
literacy_status = literacy_status.rename(columns={'Population':"Amount of Literate Population"})
literacy_status.to_csv("./DataSets/literacy_status.csv")






# %%
#exploring indigenous_status
df = pd.read_csv("./SourceDataSets/indigenous_status.csv")
indigenous_status = df[df["Indigenous Self Abscribing"] == "Yes"]
aggregation_functions = {'National Urban System': 'first', "Population":"first"}
indigenous_status = df.groupby(indigenous_status['National Urban System ID']).aggregate(aggregation_functions)
indigenous_status = indigenous_status.rename(columns={'Population':"Amount of Indigenous Population"})
indigenous_status.to_csv("./DataSets/indigenous_status.csv")









# %%
#exploring salary_wages
df = pd.read_csv("./SourceDataSets/salary_wages.csv")

#%%
df["National Urban System ID"].unique().size

#%%
#considering an average salary of 7800 MX as of 2023
salary_wages = df.loc[
    (df["Salary Group"] == "$4k - $5k") |
    (df["Salary Group"] == "$5k - $6k") |
    (df["Salary Group"] == "$6k - $7k") |
    (df["Salary Group"] == "$2k - $3k") |
    (df["Salary Group"] == "$7k - $8k") |
    (df["Salary Group"] == "< $2k")
    ]
salary_wages

#%%
#adding the workforce:
#dropped "Salary Group ID" and "Salary Group"
aggregation_functions = {'National Urban System': 'first', "Workforce":"sum"}
aggregated_salaries = df.groupby(df['National Urban System ID']).aggregate(aggregation_functions)
aggregated_salaries = aggregated_salaries.rename(columns={'Workforce':"Average Salary or Less"})
aggregated_salaries.to_csv("./DataSets/average_salaries_or_less.csv", index_col=False)



# %%
