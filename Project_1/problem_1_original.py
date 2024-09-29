# to perform EDA on the data
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

#read csv
df = pd.read_csv('heart1.csv')
print(df)

#check for null values
def check_null_values(df):
    return df.isnull().values.sum() == 0

if check_null_values(df):
    print('\n no missing nor null values in data\n')
else:
    print('\n data has missing or nul values\n')

#glance the data
df.describe()

#covariance matrix
covariance_mat = df.cov()
print(f'\n covariance matrix: \n{covariance_mat}')

#correlation matrix
correlation_mat = df.corr()
print(f'\n correlation_matrix: \n{correlation_mat}')

#to visualize histogram representation of data
df.hist(bins=30, figsize=(15,10))
plt.show()

#heat map
plt.figure(figsize=(10,8))
sns.heatmap(correlation_mat, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

#to see the pairplot to see relation among variables
sns.pairplot(df)
plt.show()

#detect outliers
df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(15,10))

#Scatter plot
df.plot.scatter(x='age', y='rbp')

#Bar plots
df['sex'].value_counts().plot(kind='bar')

# VIF Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
print(f"\n VIF Values in sorted form: \n{vif_data.sort_values(by='VIF', ascending=False)}")

# Calculate z-scores
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).sum(axis=0)
print("\n Outliers per column:\n", outliers)
