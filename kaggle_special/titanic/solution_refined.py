import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def refine_features(df):
    df = df.copy()
    # Title Extraction
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    # Sex
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Age handling by Title
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Fare
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())

    # Select features explicitly
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    return df[features]

X_train = refine_features(train)
y_train = train['Survived']
X_test = refine_features(test)

# Tuned RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission_refined.csv', index=False)
print("Refined Titanic submission created.")
