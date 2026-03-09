import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def preprocess(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in ['Id', 'SalePrice']:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.factorize(df[col].fillna('None'))[0]

    features = [c for c in df.columns if c not in ['Id', 'SalePrice']]
    return df[features]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = preprocess(train)
y_train = np.log1p(train['SalePrice'])
X_test = preprocess(test)

m1 = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
m2 = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, verbosity=-1)
m3 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)

m1.fit(X_train, y_train)
m2.fit(X_train, y_train)
m3.fit(X_train, y_train)

preds = (np.expm1(m1.predict(X_test)) + np.expm1(m2.predict(X_test)) + np.expm1(m3.predict(X_test))) / 3
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_ensemble.csv', index=False)
print("House Prices ensemble submission created.")
