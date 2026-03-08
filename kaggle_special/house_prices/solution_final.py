import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df[cols]

X_train = process(train)
y_train = np.log1p(train['SalePrice'])
X_test = process(test)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

preds = np.expm1(model.predict(X_test))
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_final.csv', index=False)
print("Final House Prices submission created.")
