import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    # High signal features only
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Reasoning-driven: Quality * Area interaction
    df['QualArea'] = df['OverallQual'] * df['GrLivArea']

    return df[cols + ['QualArea']]

X_train = process(train)
y_train = np.log1p(train['SalePrice'])
X_test = process(test)

m1 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
m2 = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

m1.fit(X_train, y_train)
m2.fit(X_train, y_train)

preds = (np.expm1(m1.predict(X_test)) + np.expm1(m2.predict(X_test))) / 2
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_ultimate.csv', index=False)
print("Ultimate House Prices submission created.")
