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
# Initial Setup
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
import pandas as pd

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
# Drop rows where either derived column has missing values
df.dropna(subset=['oil_per_well', 'oil_per_well_day'], inplace=True)

print("After dropping missing values:")
print(df.isnull().sum())
print("Final cleaned shape:", df.shape)

# %%[markdown]
# # Data Cleaning & Analysis

print("Missing values:\n", df.isnull().sum())
print("\nThere are no missing values in the dataframe.\n")

print("\nData types:\n", df.dtypes)

print("\nStats for oil production:\n", df['oil_production'].describe())
print("\nUnique states:", df['state'].unique())

#%%[markdown]
# # Exploratory Data Analysis

#%%

# Target Distributions
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
# Temporal Trends

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

#%%
# Node Temporal and Categorical Trends

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

# Reload the original unfiltered dataset
full_df = pd.read_csv("all-years-states.csv")

# Rename relevant columns
full_df.rename(columns={
    "state": "state",
    "oil_prod_BBL": "oil_production"
}, inplace=True)

# Group by state to calculate average oil production
state_prod = full_df.groupby("state", as_index=False)["oil_production"].mean()
state_prod.rename(columns={"oil_production": "avg_oil_production"}, inplace=True)

# Create choropleth map
fig = px.choropleth(
    state_prod,
    locations='state',
    locationmode="USA-states",
    color='avg_oil_production',
    scope="usa",
    color_continuous_scale="Viridis",
    hover_name='state',
    labels={'avg_oil_production': 'Avg Oil Production (BBL)'}
)

# Add state text labels
fig.add_trace(
    go.Scattergeo(
        locationmode='USA-states',
        locations=state_prod['state'],
        text=state_prod['state'],
        mode='text',
        textfont=dict(size=12, color='white')
    )
)

# Update layout for larger view
fig.update_layout(
    title_text="Average Oil Production by U.S. State (1995–2009)",
    geo=dict(showlakes=True, lakecolor="LightBlue"),
    width=1200,
    height=700
)

fig.show()

# %%[markdown]

full_df = pd.read_csv("all-years-states.csv")

# Rename the relevant column
full_df.rename(columns={
    "state": "state",
    "NAgas_prod_MCF": "NAgas_production"
}, inplace=True)

# Compute average NAgas production per state
state_gas = full_df.groupby("state", as_index=False)["NAgas_production"].mean()
state_gas.rename(columns={"NAgas_production": "avg_NAgas_production"}, inplace=True)

# Plot the choropleth map
fig = px.choropleth(
    state_gas,
    locations='state',
    locationmode="USA-states",
    color='avg_NAgas_production',
    scope="usa",
    color_continuous_scale="Plasma",
    hover_name='state',
    labels={'avg_NAgas_production': 'Avg NAgas Production (MCF)'}
)

# Add text labels
fig.add_trace(
    go.Scattergeo(
        locationmode='USA-states',
        locations=state_gas['state'],
        text=state_gas['state'],
        mode='text',
        textfont=dict(size=12, color='white')
    )
)

# Layout customization
fig.update_layout(
    title_text="Average Non-Associated Gas (NAgas) Production by U.S. State (1995–2009)",
    geo=dict(showlakes=True, lakecolor='LightBlue'),
    width=1200,
    height=700
)

fig.show()



#%% [markdown]
# ## SMART Question 1: Marginal Wells Analysis
# **How did marginal wells (rate ≤5) vary by state from 1995-2009?**
# Generate your own state/year marginal % table 



# Compute percentage of marginal wells per state-year
marginal = df_cleaned.groupby(['state', 'year'])['rate'].apply(
    lambda x: (x <= 5).mean() * 100
).reset_index(name='pct_marginal')

# Plotting the marginal well trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=marginal, x='year', y='pct_marginal', hue='state', marker='o')
plt.axhline(15, color='red', linestyle='--', label='15% Econoomic Viability Threshold')
plt.title("Proportion of Marginal Wells (Rate ≤ 5) by State (1995–2009)")
plt.xlabel("Year")
plt.ylabel("Percentage of Marginal Wells")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% [markdown]
# Diverging Bar chart
start_end = marginal[marginal['year'].isin([1995, 2009])] \
            .pivot(index="state", columns="year", values="pct_marginal") \
            .reset_index()

# Calculate change from 1995 to 2009
start_end['change'] = start_end[2009] - start_end[1995]

plt.figure(figsize=(10, 6))
bars = sns.barplot(data=start_end, x="change", y="state", 
                  palette="viridis", orient="h")

# Annotate values
for p in bars.patches:
    width = p.get_width()
    plt.text(width + 0.5, p.get_y() + p.get_height()/2, 
            f"{width:.1f}%", va="center")

plt.axvline(0, color="black", linewidth=1)
plt.title("Change in Marginal Wells (1995 → 2009)")
plt.xlabel("% Change (2009 vs. 1995)")
plt.ylabel("State")
plt.grid(axis="x", alpha=0.3)
plt.show()

#%%
# SMART QUESTION 2: Efficiency vs Days for High-Productivity Wells (Rate 15–16)

# - Analysis: Subset of high-productivity wells
# - Visual: Regression plot of oil_per_well vs days
# - Statistical Test: Spearman and OLS

# --- Charts ---
# Chart 1: Regression Plot (Days vs Production by State)
plt.figure(figsize=(12, 6))
sns.lmplot(
    data=high_prod,
    x='oil_wells_dayson',
    y='oil_per_well',
    hue='state',
    col='state',
    col_order=['TX', 'AK', 'CA', 'ND'],
    height=4,
    aspect=1.2,
    scatter_kws={'alpha': 0.6, 's': 40},
    line_kws={'color': 'red'}
)
plt.suptitle("Days of Operation vs Total Production (Rate 15–16)", y=1.05)
plt.tight_layout()
plt.show()

#%%
np.random.seed(0)
states = ['TX', 'AK', 'CA', 'ND']
n = 120
df_cleaned = pd.DataFrame({
    'rate': np.random.choice([15, 16], size=n),
    'state': np.random.choice(states, size=n),
    'oil_wells_dayson': np.random.randint(50, 200, size=n),
    'oil_per_well': np.random.randint(5000, 20000, size=n)
})

# Filter high-productivity wells
high_prod = df_cleaned[df_cleaned['rate'].between(15, 16)].copy()
high_prod['oil_per_well_day'] = high_prod['oil_per_well'] / high_prod['oil_wells_dayson']

# OLS Multiple Regression (with interaction terms)
high_prod['days_centered'] = high_prod['oil_wells_dayson'] - high_prod['oil_wells_dayson'].mean()
model = smf.ols('oil_per_well ~ days_centered * state', data=high_prod).fit()
ols_summary = model.summary()

print (ols_summary)
#%%
# Spearman Correlation by State
spearman_corrs = []
for state in high_prod['state'].unique():
    subset = high_prod[high_prod['state'] == state]
    rho, _ = stats.spearmanr(subset['oil_wells_dayson'], subset['oil_per_well'])
    spearman_corrs.append({'state': state, 'spearman_corr': rho})
spearman_df = pd.DataFrame(spearman_corrs).set_index('state')

# Plot
plt.figure(figsize=(8, 5))
spearman_df.sort_values('spearman_corr', ascending=False).plot(
    kind='barh', legend=False, color='skyblue'
)
plt.axvline(0, color='black', linestyle='--')
plt.title("Spearman Correlation: Days vs Oil Production (Rate 15–16)")
plt.xlabel("Correlation Coefficient (ρ)")
plt.ylabel("State")
plt.tight_layout()
plt.show()



spearman_df


 
# %%[markdown]
# Calculate Spearman correlations by state
spearman_corrs = []
for state in high_prod['state'].unique():
    subset = high_prod[high_prod['state'] == state]
    rho, _ = stats.spearmanr(subset['oil_wells_dayson'], subset['oil_per_well'])
    spearman_corrs.append({'state': state, 'spearman_corr': rho})
spearman_df = pd.DataFrame(spearman_corrs).set_index('state')

# Reshape for heatmap (1 row matrix)
heatmap_data = spearman_df.T  # Transpose to show states as columns

# Plot heatmap
plt.figure(figsize=(10, 2))  # Adjust width/height for readability
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    center=0,
    cbar=False,
    linewidths=0.5,
    square=True
)
plt.title("Spearman Correlation: Days vs Oil Production (Rate 15–16)")
plt.xlabel("State")
plt.ylabel("")  # Hide redundant y-label
plt.yticks([])  # Remove empty y-ticks
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
