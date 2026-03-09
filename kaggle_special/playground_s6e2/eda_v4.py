import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
train['Target'] = (train['Heart Disease'] == 'Presence').astype(int)

# Identify why 0.95+ is possible
print("Exploring high-cardinality interactions...")

# Interaction: Thallium * Vessels * Chest Pain
train['Thal_Vess_Chest'] = (train['Thallium'].astype(str) + "_" +
                            train['Number of vessels fluro'].astype(str) + "_" +
                            train['Chest pain type'].astype(str))

counts = train['Thal_Vess_Chest'].value_counts()
means = train.groupby('Thal_Vess_Chest')['Target'].mean()

print("\nThal_Vess_Chest Decision Purity (Top 20):")
print(means[counts > 100].sort_values(ascending=False).head(20))

# Interaction: ST_Slope * Sex
train['ST_Slope_Sex'] = (train['ST depression'].astype(str) + "_" +
                         train['Slope of ST'].astype(str) + "_" +
                         train['Sex'].astype(str))

means_st = train.groupby('ST_Slope_Sex')['Target'].mean()
counts_st = train['ST_Slope_Sex'].value_counts()

print("\nST_Slope_Sex Decision Purity (Top 20):")
print(means_st[counts_st > 100].sort_values(ascending=False).head(20))
