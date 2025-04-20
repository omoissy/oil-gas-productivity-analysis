# %% 
# %% 
# Oil and Gas Well Productivity Analysis (1995–2009)

# Node 1: Setup and Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import plotly.express as px

import os
os.chdir('/Users/mac/OIL-GAS-PRODUCTIVITY-ANALYSIS/data/raw')
# %%
# Load and prepare original dataset
original_df = pd.read_excel('all-years-states.xls')

# Create a working copy for analysis
df = original_df.copy()

# Feature engineering on original data
df['oil_per_well'] = df['oil_prod_BBL'] / df['num_oil_wells']
df['oil_per_well'].fillna(df['oil_per_well'].median(), inplace=True)
# %%

# ====================
# 1. Basic Data Inspection
# ====================
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
display(df.head())

print("\nColumn Dtypes:")
print(df.dtypes)

print("\nBasic Description:")
display(df.describe(include='all').T)

# %%
# ====================
# 2. Missing Values Analysis
# ====================
print("\nMissing Values Summary:")
missing = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Missing %': round(df.isnull().mean()*100, 2)
})
display(missing[missing['Missing Values'] > 0])

# %%
# ====================
# 3. Data Distributions
# ====================
# Numerical variables distribution
num_cols = ['oil_prod_BBL', 'NAgas_prod_MCF', 'oil_wells_dayson', 
            'gas_wells_dayson', 'conden_prod_BBL']

plt.figure(figsize=(18, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(np.log1p(df[col]), kde=True)
    plt.title(f'Distribution of {col} (log scale)')
    plt.xlabel('')
plt.tight_layout()
plt.show()

# %%
# Categorical variables distribution
cat_cols = ['state', 'rate_class']

plt.figure(figsize=(18, 6))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# ====================
# 4. Temporal Analysis
# ====================

print(df.columns.tolist())
# %%

# Create filtered dataset for visualization only
states_to_plot = ['TX', 'AK', 'CO', 'NM', 'WY', 'ND', 'CA', 'OK', 'LA', 'WV']
plot_df = df[df['state'].isin(states_to_plot)].copy()

plt.figure(figsize=(16, 8))
sns.lineplot(
    data=plot_df, 
    x='prod_year', 
    y='oil_prod_BBL', 
    hue='state',
    estimator='sum',
    errorbar=None,
    palette='tab20'
)
plt.title('Total Oil Production by State (1995–2009)', fontsize=14)
plt.xlabel('Production Year', fontsize=12)
plt.ylabel('Oil Production (Barrels)', fontsize=12)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# ====================
# 5. Correlation Analysis
# ====================
# Calculate correlations
print(df.columns.tolist())
# %%

# Use original dataset for correlations
corr_cols = [
    'oil_prod_BBL',
    'NAgas_prod_MCF',
    'oil_wells_dayson',
    'gas_wells_dayson',
    'conden_prod_BBL',
    'num_oil_wells',
    'num_gas_wells'
]

corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
           mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# %%
# ====================
# 6. State-wise Analysis
# ====================
# Top 10 states by oil production
top_states = df.groupby('state')['oil_prod_BBL'].sum().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')
plt.title('Top 10 States by Total Oil Production')
plt.xlabel('Total Oil Production (BBL)')
plt.ylabel('State')
plt.show()

# %%
# ====================
# 7. Rate Class Analysis
# ====================
# Oil production by rate class
rate_class_analysis = df.groupby('rate_class').agg({
    'oil_prod_BBL': 'sum',
    'num_oil_wells': 'sum'
}).reset_index()

plt.figure(figsize=(14, 6))
sns.scatterplot(data=rate_class_analysis, x='rate_class', y='oil_prod_BBL',
                size='num_oil_wells', sizes=(50, 500))
plt.title('Oil Production vs. Rate Class')
plt.xlabel('Rate Class')
plt.ylabel('Total Oil Production (BBL)')
plt.show()

# %%
# ====================
# 8. Outlier Detection
# ====================
# Z-score analysis for outliers
z_scores = stats.zscore(df[num_cols])
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores > 3).any(axis=1)
print(f"\nNumber of potential outliers: {outliers.sum()}")

# %%
# ====================
# 9. Feature Engineering Checks
# ====================
# Create productivity metrics
df['oil_per_well'] = df['oil_prod_BBL'] / df['num_oil_wells']
df['days_per_oil_well'] = df['oil_wells_dayson'] / df['num_oil_wells']

# Check new features
print("\nNew Features Summary:")
display(df[['oil_per_well', 'days_per_oil_well']].describe())

# %%
# ====================
# 10. Advanced Visualizations
# ====================
# Pairplot for key variables
sns.pairplot(df[['oil_prod_BBL', 'NAgas_prod_MCF', 'oil_wells_dayson', 
                'rate_class']], hue='rate_class', palette='viridis')
plt.suptitle('Pairwise Relationships with Rate Class', y=1.02)
plt.show()

# Boxplot of oil production by rate class
plt.figure(figsize=(16, 8))
sns.boxplot(data=df, x='rate_class', y='oil_prod_BBL', showfliers=False)
plt.title('Oil Production Distribution by Rate Class')
plt.xlabel('Rate Class')
plt.ylabel('Oil Production (BBL)')
plt.show()
# %%
# Filter to selected states
selected_states = ['TX', 'AK', 'CO', 'NM', 'ND', 'CA']
filtered_df = df[df['state'].isin(selected_states)]

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=filtered_df, 
    x='rate_class', 
    y='oil_prod_BBL', 
    hue='state',  # Add state distinction
    showfliers=False
)
plt.title('Oil Production by Rate Class (Key States)')
plt.xlabel('Rate Class')
plt.ylabel('Oil Production (BBL)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# %%
# ====================
# 11. Temporal Trends by Rate Class
# ====================
# Marginal wells (rate class <=5) trend
marginal_wells = df[df['rate_class'] <= 5].groupby(['prod_year', 'state']).agg({
    'num_oil_wells': 'sum'
}).reset_index()

plt.figure(figsize=(16, 8))
sns.lineplot(data=marginal_wells, x='prod_year', y='num_oil_wells', hue='state')
plt.title('Marginal Wells (Rate Class ≤5) Trend by State')
plt.xlabel('Year')
plt.ylabel('Number of Marginal Wells')
plt.show()

# %%
marginal_wells_filtered = filtered_df[filtered_df['rate_class'] <= 5].groupby(
    ['prod_year', 'state']
).agg({'num_oil_wells': 'sum'}).reset_index()

plt.figure(figsize=(16, 8))
sns.lineplot(
    data=marginal_wells_filtered,
    x='prod_year', 
    y='num_oil_wells', 
    hue='state',
    style='state',  # Differentiate lines
    markers=True,   # Add markers
    dashes=False
)
plt.title('Marginal Wells (Rate Class ≤5) Trend - Key States')
plt.xlabel('Year')
plt.ylabel('Number of Marginal Wells')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
