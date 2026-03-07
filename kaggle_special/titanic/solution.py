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
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # Encode categorical
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df[features], df.get('PassengerId')

X_train, _ = preprocess(train_data)
y_train = train_data['Survived']
X_test, passenger_ids = preprocess(test_data)

# Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Submit
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Titanic submission.csv created.")
