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

#%%
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
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv('all-years-states.csv')
df.drop(columns=["NAgas_prod_MCF", "conden_prod_BBL"], inplace=True)

df.rename(columns={
    "state": "state",
    "prod_year": "year",
    "rate_class": "rate",
    "num_oil_wells": "oil_wells_count",
    "oil_prod_BBL": "oil_production",
    "ADgas_prod_MCF": "gas_production",
    "oil_wells_dayson": "oil_wells_dayson",
    "num_gas_wells": "gas_wells_count",
    "gas_wells_dayson": "gas_wells_dayson"
}, inplace=True)


print("Structure of the dataframe is - \n" ,df.info())
print("\nHere are some values for reference - \n" ,df.head())

# %%[markdown]
# # Data Cleaning & Analysis

print("Missing values:\n", df.isnull().sum())
print("\nThere are no missing values in the dataframe.\n")

print("\nData types:\n", df.dtypes)

print("\nStats for oil production:\n", df['oil_production'].describe())
print("\nUnique states:", df['state'].unique())

# %%

sns.histplot(df['oil_production'], bins=30, color='skyblue')
plt.title('Oil Production Histogram')
plt.xlabel('Oil Production (BBL)')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df['gas_production'], bins=30, color='skyblue')
plt.title('Gas Production Histogram')
plt.xlabel('Gas Production (BBL)')
plt.ylabel('Frequency')
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
