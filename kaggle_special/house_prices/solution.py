import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing
def preprocess(df):
    df = df.copy()
    # Only use numeric features for simplicity in this baseline
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Drop Id and target if they exist
    features = [col for col in numeric_df.columns if col not in ['Id', 'SalePrice']]
    return numeric_df[features], df.get('Id')

X_train, _ = preprocess(train_data)
y_train = np.log1p(train_data['SalePrice']) # Log transform target
X_test, house_ids = preprocess(test_data)

# Model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
model.fit(X_train, y_train)

# Predict
predictions = np.expm1(model.predict(X_test))

# Submit
output = pd.DataFrame({'Id': house_ids, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
print("House Prices submission.csv created.")
