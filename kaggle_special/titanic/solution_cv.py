import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = pd.factorize(df['Embarked'])[0]

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    return df[features]

X_train = process(train)
y_train = train['Survived']
X_test = process(test)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=10)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

model.fit(X_train, y_train)
preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}).to_csv('submission_cv.csv', index=False)
