# To add a new cell, type ''
# To add a new markdown cell, type '#%%[markdown]'

#%%[markdown]
#
# # Oil Gas Productivity Analysis
# ## By: Team 3
#
# 
# The variables in the dataset are:  
# * state: U.S. state abbreviation (e.g., AK for Alaska)
# * prod_year: Production year of the recorded data
# * rate_class: Rate class category 
# * num_oil_wells: Number of active oil wells
# * oil_prod_BBL: Oil production in barrels (BBL)
# * oil_wells_dayson: Total days of oil well production activity
# * num_gas_wells: Number of active gas wells
# * NAgas_prod_MCF: Non-associated gas production in thousand cubic feet (MCF)
# * conden_prod_BBL: Condensate production in barrels (BBL)
# * gas_wells_dayson: Total days of gas well production activity

# ### Analysis Tasks:
# 1. Predict Oil Production (Regression)
# 2. Predict Non-Associated Gas Production (Regression)
# 3. Predict Rate Class (Classification)

#%%
# ====================
# Initial Setup
# ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, 
                             accuracy_score, confusion_matrix, 
                             classification_report)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#%%
# Load and prepare data

# Load dataset
df = pd.read_csv('all-years-states.csv')

# Rename columns for consistency
df.rename(columns={
    "state": "state",
    "prod_year": "year",
    "rate_class": "rate",
    "num_oil_wells": "oil_wells_count",
    "oil_prod_BBL": "oil_production",
    "ADgas_prod_MCF": "ADgas_production",
    "oil_wells_dayson": "oil_wells_dayson",
    "num_gas_wells": "gas_wells_count",
    "NAgas_prod_MCF": "NAgas_production",
    "conden_prod_BBL": "condensate_production",
    "gas_wells_dayson": "gas_wells_dayson"
}, inplace=True)

# Filter the data for years 1995-2009 and key states
df = df[
    (df['year'].between(1995, 2009)) &
    (df['state'].isin(['TX', 'AK', 'CA', 'ND']))
].drop(columns=['ADgas_production', 'condensate_production'])

# Create productivity metrics
df['oil_per_well'] = df['oil_production'] / df['oil_wells_count']
df['oil_per_well_day'] = df['oil_production'] / df['oil_wells_dayson']

# Drop rows with missing values
df_cleaned = df.dropna()

# Display the cleaned DataFrame info and first few rows
df_shape = df_cleaned.shape
df_range = (df_cleaned['year'].min(), df_cleaned['year'].max())
df_info = df_cleaned.info()
df_head = df_cleaned.head()

(df_shape, df_range, df_head)


# %%[markdown]
# # Data Cleaning & Analysis
# Drop rows where either derived column has missing values
df.dropna(subset=['oil_per_well', 'oil_per_well_day'], inplace=True)

print("After dropping missing values:")
print(df.isnull().sum())
print("Final cleaned shape:", df.shape)


# %%[markdown]
print("Missing values:\n", df.isnull().sum())
print("\nThere are no missing values in the dataframe.\n")

print("\nData types:\n", df.dtypes)

print("\nStats for oil production:\n", df['oil_production'].describe())
print("\nUnique states:", df['state'].unique())

#%%[markdown]
# # Exploratory Data Analysis

#%%
# ====================
# Target Distributions
# ====================
plt.figure(figsize=(18,5))

# Oil Production
plt.subplot(1,3,1)
sns.histplot(np.log1p(df['oil_production']), kde=True, color='teal')
plt.title('Oil Production (Log Scale)')

# NAgas Production
plt.subplot(1,3,2)
sns.histplot(np.log1p(df['NAgas_production']), kde=True, color='orange')
plt.title('Non-Associated Gas (Log Scale)')

# Rate Class
plt.subplot(1,3,3)
sns.countplot(x='rate', data=df, palette='viridis')
plt.title('Rate Class Distribution')

plt.tight_layout()
plt.show()



#%%
# ====================
# Temporal Trends
# ====================
plt.figure(figsize=(16,6))

# Oil Production Trend
sns.lineplot(data=df, x='year', y='oil_production', 
            estimator='median', errorbar=None,
            color='teal', label='Oil')
plt.title('National Production Trends 1995-2009')
plt.xlabel('Year')
plt.ylabel('Median Production')

# NAgas Production Trend
sns.lineplot(data=df, x='year', y='NAgas_production', 
            estimator='median', errorbar=None,
            color='orange', label='Non-Associated Gas')

plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %% 

# ====================
# Node 4: Temporal and Categorical Trends
# ====================
# ====================
# Node 4: Temporal and Categorical Trends (Fixed)
# ====================
plt.figure(figsize=(16, 8))
sns.lineplot(
    data=df[df['state'].isin(['TX', 'AK'])],
    x='year',
    y='oil_production',
    hue='state',
    estimator='sum',
    errorbar=None
)
plt.title('Oil Production in TX vs AK')
plt.xlabel('Year')
plt.ylabel('Total Oil Production (BBL)')
plt.grid(alpha=0.3)
plt.show()

#%%
# ====================
# Feature Correlations
# ====================
corr_matrix = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
           vmin=-1, vmax=1, mask=np.triu(np.ones_like(corr_matrix)))
plt.title('Feature Correlation Matrix')
plt.show()

# %%[markdown]
# Oil production by state comparision

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='state', y='oil_production')
plt.title('Oil Production by State')
plt.xlabel('State')
plt.ylabel('Oil Production')
plt.xticks(rotation=45)
plt.show()


groups = [group['oil_production'].values for name, group in df.groupby('state')]

f_stat, p_value = stats.f_oneway(*groups)

print(f'ANOVA F-statistic: {f_stat}')
print(f'p-value: {p_value}')

if p_value < 0.05:
    print("There is a statistically significant difference in oil production among states.")
else:
    print("No statistically significant difference in oil production among states.")

# %%[markdown]
# Oil production & days of oil well production 
from scipy.stats import pearsonr

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='oil_wells_dayson', y='oil_production', alpha=0.5)
plt.title('Oil Production vs. Days of Oil Well Production')
plt.xlabel('Days of Oil Well Production')
plt.ylabel('Oil Production')
plt.show()


corr, p_value = pearsonr(df['oil_wells_dayson'], df['oil_production'])
print(f"Pearson correlation coefficient: {corr}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a statistically significant linear relationship between oil production and days of oil well production.")
else:
    print("No statistically significant linear relationship detected.")
    
 
# %%[markdown]
# Correlation Plot of Oil Production Variables

corr = df.select_dtypes(include=['int64']).corr()
plt.figure(figsize=(12, 8))

sns.heatmap(
    corr, 
    annot=True,      
    fmt=".2f",       
    cmap='coolwarm', 
    vmin=-1, vmax=1, 
    linewidths=0.5,
    square=True
)

plt.title("Correlation Matrix of Oil Production Variables", pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right')  
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


#%%
# ====================
# Task 1: Oil Production Prediction (Regression)
# ====================
# Features: oil_wells_count, oil_wells_dayson, ADgas_production
# Target: oil_production

# Prepare data
X = df[['oil_wells_count', 'oil_wells_dayson', 'ADgas_production']]
y = np.log1p(df['oil_production'])  # Log transform for better performance

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
oil_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and predict
oil_pipe.fit(X_train, y_train)
preds = oil_pipe.predict(X_test)

# Evaluate
print("\nTask 1 - Oil Production Prediction:")
print(f"R2 Score: {r2_score(y_test, preds):.3f}")
print(f"MAE: {mean_absolute_error(np.expm1(y_test), np.expm1(preds)):,.0f} barrels")

#%%
# ====================
# Task 2: NAgas Production Prediction (Regression)
# ====================
# Features: gas_wells_count, gas_wells_dayson, condensate_production
# Target: NAgas_production

# Prepare data
X = df[['gas_wells_count', 'gas_wells_dayson', 'condensate_production']]
y = np.log1p(df['NAgas_production'])  # Log transform

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
gas_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and predict
gas_pipe.fit(X_train, y_train)
preds = gas_pipe.predict(X_test)

# Evaluate
print("\nTask 2 - NAgas Production Prediction:")
print(f"R2 Score: {r2_score(y_test, preds):.3f}")
print(f"MAE: {mean_absolute_error(np.expm1(y_test), np.expm1(preds)):,.0f} MCF")

#%%
# ====================
# Task 3: Rate Class Prediction (Classification)
# ====================


#%%
# Task 3: Rate Class Prediction (Classification)
# Drop irrelevant columns and define features/target

# Features: All except state and year
# Target: rate



X = df.drop(columns=['state', 'year', 'rate'])
y = df['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
clf_pipe.fit(X_train, y_train)

# Predict
y_pred = clf_pipe.predict(X_test)

# Classification report as dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Show report
print("\nClassification Report (Rate Class):")
print(report_df)

#%% 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title("Confusion Matrix - Rate Class Prediction", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance in Rate Class Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %%%%[markdown]
# Oil Production Prediction -  Random forest

X = df.drop('oil_production', axis=1)
y = df['oil_production']

numeric_features = ['year', 'rate', 'oil_wells_count', 'gas_production', 
                   'oil_wells_dayson', 'gas_wells_count', 'gas_wells_dayson']
categorical_features = ['state']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=42)

rfe = RFE(estimator=model, n_features_to_select=5, step=1)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', rfe),
    ('regressor', model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.0f} barrels")

# %%[markdown]
# Recursive Feature Elimination

feature_names = (numeric_features + 
                list(pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

selected_features = np.array(feature_names)[pipeline.named_steps['feature_selection'].support_]

top_rank = []
top_rank_name = []
for i, (name, rank) in enumerate(zip(feature_names, 
                                   pipeline.named_steps['feature_selection'].ranking_)):
    print(f"{i+1}. {name}: {rank}")
    if rank < 10 :
        top_rank.append(rank)
        top_rank_name.append(name)
    


plt.barh(top_rank_name, top_rank)  
plt.title("Feature Importance (Lower Rank = More Important)")
plt.show()    

# %%[markdown]
# Model Performance after Feature Elimination

print("The selected features in RFE were ",selected_features)

X_train_selected = pipeline.named_steps['feature_selection'].transform(
    pipeline.named_steps['preprocessor'].transform(X_train)
)
X_test_selected = pipeline.named_steps['feature_selection'].transform(
    pipeline.named_steps['preprocessor'].transform(X_test)
)

model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

print(f"R2: {r2_score(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)} barrels")
# %%
