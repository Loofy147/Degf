import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

def get_titles(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    return df

def feature_eng(df, train_fare_median):
    df = df.copy()
    df = get_titles(df)
    df['Fare'] = df['Fare'].fillna(train_fare_median)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5, labels=False)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    df['Embarked'] = pd.factorize(df['Embarked'].fillna('S'))[0]
    df['Deck'] = pd.factorize(df['Cabin'].str.get(0).fillna('U'))[0]
    df['Title'] = pd.factorize(df['Title'])[0]

    # Feature selection - MUST BE IDENTICAL for Train and Test
    cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'Deck', 'FareBin', 'AgeBin']
    return df[cols]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = feature_eng(train, train['Fare'].median())
y_train = train['Survived']
X_test = feature_eng(test, train['Fare'].median())

base_models = [
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, random_state=42, verbosity=-1)),
    ('cat', CatBoostClassifier(n_estimators=500, depth=4, verbose=0, random_state=42))
]

stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
stack.fit(X_train, y_train)

preds = stack.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}).to_csv('submission_stack.csv', index=False)
print("Titanic Stacking Submission Created.")
