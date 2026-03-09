import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    # Log transform skew features
    df['GrLivArea'] = np.log1p(df['GrLivArea'])

    # Drop IDs
    df = df.drop(['Id'], axis=1)

    # Impute Numeric
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if c != 'SalePrice':
            df[c] = df[c].fillna(df[c].median())

    # Factorize Categorical
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = pd.factorize(df[c].fillna('None'))[0]

    return df

train_df = process(train)
y = np.log1p(train['SalePrice'])
X = train_df.drop(['SalePrice'], axis=1)
X_test = process(test)

# Ensemble: Lasso + GBR
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42).fit(X, y)

preds = (np.expm1(lasso.predict(X_test_scaled)) + np.expm1(gbr.predict(X_test))) / 2
pd.DataFrame({'Id': test.Id, 'SalePrice': preds}).to_csv('submission_top10.csv', index=False)
print("House Prices Top 10 Submission Created.")
