import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def feature_eng(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in num_cols: num_cols.remove('SalePrice')
    if 'Id' in num_cols: num_cols.remove('Id')

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Elite interactions
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Age'] = 2025 - df['YearBuilt']
    df['QualitySF'] = df['OverallQual'] * df['TotalSF']

    # Categorical
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = pd.factorize(df[c].fillna('U'))[0]

    return df

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X = feature_eng(train)
    y = np.log1p(train['SalePrice'])
    X_test = feature_eng(test)

    features = [c for c in X.columns if c not in ['Id', 'SalePrice']]

    m1 = XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.03, random_state=42)
    m1.fit(X[features], y)
    p1 = m1.predict(X_test[features])

    m2 = CatBoostRegressor(iterations=1000, depth=5, learning_rate=0.03, verbose=0, random_state=42)
    m2.fit(X[features], y)
    p2 = m2.predict(X_test[features])

    final_p = np.expm1((p1 + p2) / 2)
    pd.DataFrame({'Id': test.Id, 'SalePrice': final_p}).to_csv('submission_elite.csv', index=False)
    print("Elite House Prices Submission Created.")

if __name__ == "__main__":
    main()
