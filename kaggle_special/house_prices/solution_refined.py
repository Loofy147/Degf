import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def refine_house(df):
    df = df.copy()
    # Handle Numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Id', 'SalePrice']:
            df[col] = df[col].fillna(df[col].median())

    # Simple categorical handling
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('None')
        df[col] = pd.factorize(df[col])[0]

    features = [c for c in df.columns if c not in ['Id', 'SalePrice']]
    return df[features]

X_train = refine_house(train)
y_train = np.log1p(train['SalePrice'])
X_test = refine_house(test)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

preds = np.expm1(model.predict(X_test))
output = pd.DataFrame({'Id': test.Id, 'SalePrice': preds})
output.to_csv('submission_refined.csv', index=False)
print("Refined House Prices submission created.")
