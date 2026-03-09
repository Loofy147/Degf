import pandas as pd
import numpy as np
from xgboost import XGBRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    # Numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in num_cols: num_cols.remove('SalePrice')
    if 'Id' in num_cols: num_cols.remove('Id')

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # G-grounding interaction: Quality * Area * Age
    df['QualityAgeInteraction'] = df['OverallQual'] * df['GrLivArea'] / (2025 - df['YearBuilt'])

    return df[num_cols + ['QualityAgeInteraction']]

X_train = process(train)
y_train = np.log1p(train['SalePrice'])
X_test = process(test)

model = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

preds = np.expm1(model.predict(X_test))
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_master.csv', index=False)
print("Master House Prices Logic Created.")
