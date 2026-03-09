import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def domain_eng(df):
    df = df.copy()
    # High-signal interaction features
    df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'].fillna(0) + df['2ndFlrSF'].fillna(0)
    df['QualAge'] = df['OverallQual'] * (df['YrSold'] - df['YearBuilt'] + 1)
    df['TotalBath'] = df['FullBath'].fillna(0) + (0.5 * df['HalfBath'].fillna(0))

    # Select subset of robust features
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'QualAge', 'TotalBath']
    for c in cols:
        df[c] = df[c].fillna(df[c].median())
    return df[cols]

X_train = domain_eng(train)
y_train = np.log1p(train['SalePrice'])
X_test = domain_eng(test)

model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

preds = np.expm1(model.predict(X_test))
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_domain.csv', index=False)
print("Domain Reasoner Created.")
