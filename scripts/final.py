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

print(df['oil_production'].head())

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



# %%

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

# %%
