import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing
def preprocess(df):
    df = df.copy()
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['Destination'] = df['Destination'].fillna('TRAPPIST-1e')
    df['VIP'] = df['VIP'].fillna(False)

    # Encode categorical
    le = LabelEncoder()
    df['HomePlanet'] = le.fit_transform(df['HomePlanet'].astype(str))
    df['CryoSleep'] = le.fit_transform(df['CryoSleep'].astype(str))
    df['Destination'] = le.fit_transform(df['Destination'].astype(str))
    df['VIP'] = le.fit_transform(df['VIP'].astype(str))

    # Select features
    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP']
    return df[features], df.get('PassengerId')

X_train, _ = preprocess(train_data)
y_train = train_data['Transported']
X_test, passenger_ids = preprocess(test_data)

# Model
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Submit
output = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions})
output.to_csv('submission.csv', index=False)
print("Spaceship Titanic submission.csv created.")
