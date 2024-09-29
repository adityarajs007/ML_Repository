############# Author Info ###################################
# Adityaraj Suresh
# EEE 591
# Project1- Problem 1
# asures56@asu.edu
#############################################################

# Import necessary libraries
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv('heart1.csv')
print(df)

# Check for null values
def check_null_values(df):
    return df.isnull().sum().sum() == 0

if check_null_values(df):
    print('\nNo missing or null values in the data\n')
else:
    print('\nData contains missing or null values\n')

# Display basic statistics for numerical features
print("\nData Summary:\n", df.describe())

# Covariance matrix
covariance_mat = df.cov()
print(f'\nCovariance Matrix:\n{covariance_mat}')

# Correlation matrix
correlation_mat = df.corr()
print(f'\nCorrelation Matrix:\n{correlation_mat}')

# Visualize data distribution with histograms
df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms for All Variables', fontsize=16)
plt.show()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_mat, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)
plt.show()

# Pairplot to visualize relationships among variables
sns.pairplot(df)
plt.suptitle('Pairplot of Variables', y=1.02, fontsize=16)
plt.show()

# Detect outliers using boxplots
df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(15, 10), title='Boxplots for Outlier Detection')
plt.show()

# Scatter plot for two specific features (e.g., 'age' and 'rbp')
df.plot.scatter(x='age', y='rbp', title='Scatter Plot: Age vs RBP')
plt.show()

# Bar plot for categorical variables (e.g., 'sex')
df['sex'].value_counts().plot(kind='bar', title='Bar Plot: Sex Distribution')
plt.show()

# Variance Inflation Factor (VIF) to detect multicollinearity
vif_data = pd.DataFrame()
vif_data['feature'] = df.columns
vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
print(f"\nVIF Values in sorted form:\n{vif_data.sort_values(by='VIF', ascending=False)}")

# Calculate z-scores for outlier detection
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).sum(axis=0)
print("\nOutliers per column:\n", outliers)
