# Oil and Gas Well Productivity Analysis (1995–2009)
# FINAL INTEGRATED NOTEBOOK
# %% 

# ====================
# Node 1: Setup and Import Libraries
# ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")
# %% 

# ====================
# Node 2: Data Loading & Preprocessing
# ====================
df = pd.read_excel('all-years-states.xls')
df.columns = df.columns.str.strip().str.lower()
df['oil_per_well'] = df['oil_prod_bbl'] / df['num_oil_wells']
df['oil_per_well'].fillna(df['oil_per_well'].median(), inplace=True)
df['days_per_oil_well'] = df['oil_wells_dayson'] / df['num_oil_wells']
df['days_per_oil_well'].fillna(0, inplace=True)
# %% 

# ====================
# Node 3: Exploratory Data Analysis (EDA)
# ====================
print("Dataset shape:", df.shape)
print(df.describe())

# Distribution plots
num_cols = ['oil_prod_bbl', 'nagas_prod_mcf', 'oil_wells_dayson', 'gas_wells_dayson', 'conden_prod_bbl']
plt.figure(figsize=(18, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(np.log1p(df[col]), kde=True)
    plt.title(f'Distribution of {col} (log scale)')
plt.tight_layout()
plt.show()
# %% 

# ====================
# Node 4: Temporal and Categorical Trends
# ====================
plt.figure(figsize=(16, 8))
sns.lineplot(data=df[df['state'].isin(['TX', 'AK'])], x='prod_year', y='oil_prod_bbl', hue='state', estimator='sum')
plt.title('Oil Production in TX vs AK')
plt.show()
# %% 

# ====================
# Node 5: Correlation Analysis
# ====================
corr_cols = ['oil_prod_bbl', 'nagas_prod_mcf', 'oil_wells_dayson', 'gas_wells_dayson', 'conden_prod_bbl']
corr_matrix = df[corr_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# %% 

# ====================
# Node 6: Clustering (KMeans + PCA)
# ====================
features = ['oil_prod_bbl', 'nagas_prod_mcf', 'oil_wells_dayson', 'gas_wells_dayson', 'conden_prod_bbl']
X_scaled = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA Visual
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['cluster'], palette='viridis')
plt.title('K-Means Clusters on PCA Projection')
plt.show()
# %% 

# ====================
# Node 7: Classification - Marginal Wells
# ====================
df['high_marginal'] = (df['rate_class'] <= 5).astype(int)
X_clf = df[features]
y_clf = df['high_marginal']
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))
# %% 

# ====================
# Node 8: Regression - Predicting Oil Production
# ====================
X_reg = df[['num_oil_wells', 'oil_wells_dayson', 'nagas_prod_mcf']].fillna(0)
y_reg = df['oil_prod_bbl']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
print("\nLinear Regression R^2:", r2_score(y_test_reg, y_pred_reg))
# %% 

# ====================
# Node 9: Association Rule Mining
# ====================
assoc_df = pd.DataFrame()
assoc_df['high_marginal'] = df['high_marginal']
assoc_df['high_gas'] = df['nagas_prod_mcf'] > df['nagas_prod_mcf'].median()
assoc_df['high_oil'] = df['oil_prod_bbl'] > df['oil_prod_bbl'].median()
assoc_df = assoc_df.astype(int)
frequent_itemsets = apriori(assoc_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
# %% 

# ====================
# Node 10: Visual Insights
# ====================
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='cluster', palette='viridis')
plt.title('Cluster Distribution')
plt.show()

plt.figure(figsize=(16, 8))
sns.lineplot(data=df, x='prod_year', y='oil_prod_bbl', hue='cluster', estimator='median')
plt.title('Oil Production Trends by Cluster')
plt.show()

state_prod = df.groupby('state', as_index=False).agg({'oil_prod_bbl': 'mean', 'nagas_prod_mcf': 'median'})
fig = px.choropleth(
    state_prod,
    locations='state',
    locationmode="USA-states",
    color='oil_prod_bbl',
    hover_name='state',
    hover_data=['nagas_prod_mcf'],
    scope="usa",
    color_continuous_scale="Viridis",
    title='State-Level Oil Production (BBL)'
)
fig.show()
# %% 

# ====================
# Save Processed Data
# ====================
df.to_csv('processed_oil_well_data.csv', index=False)
print("\nData pipeline completed and saved.")

