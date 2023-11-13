#%%
#imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sb

#imports for metrics evaluation
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import LabelEncoder

#imports for models
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVC

#%%
DF = pd.read_csv("./df_normalized.csv")
_FEATURE_COLUMNS = DF[["indigenous", "Poverty", "higher_education", "healthcare","age_15_29", "age_30_44", "age_45_59", "age_60_75", "catholic", "foreign_migrant"]]

#%%
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

# %%
boxplot()

# %%
