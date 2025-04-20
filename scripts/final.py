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
# * rate_class: Rate class category (1 to 7)
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
import rfit 


df = pd.read_csv('all-years-states.csv')

print(df.info())
print("\n")
print(df.head())

 
# %%
