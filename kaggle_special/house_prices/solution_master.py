import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def refine_house(df):
    df = df.copy()
    # High-reasoning features
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['Has2ndFlr'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # Impute and factorize
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Id', 'SalePrice']:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col].fillna('None'))[0]

    features = [c for c in df.columns if c not in ['Id', 'SalePrice']]
    return df[features]

X_train = refine_house(train)
y_train = np.log1p(train['SalePrice'])
X_test = refine_house(test)

m1 = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42)
m2 = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42, verbosity=-1)

m1.fit(X_train, y_train)
m2.fit(X_train, y_train)

preds = (np.expm1(m1.predict(X_test)) + np.expm1(m2.predict(X_test))) / 2
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_master.csv', index=False)
print("House Prices Master Submission Created.")
