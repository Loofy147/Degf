import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def feature_eng(df):
    df = df.copy()
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Fare'] = df['Fare'].fillna(32.2)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Age'] < 12)).astype(int)

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = pd.factorize(df['Embarked'].fillna('S'))[0]
    df['Title'] = pd.factorize(df['Title'])[0]

    return df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsWomanOrChild']]

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X = feature_eng(train)
    y = train['Survived']
    X_test = feature_eng(test)

    m1 = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, random_state=42)
    m1.fit(X, y)
    p1 = m1.predict_proba(X_test)[:, 1]

    m2 = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.01, verbose=0, random_state=42)
    m2.fit(X, y)
    p2 = m2.predict_proba(X_test)[:, 1]

    final_p = (p1 + p2) / 2
    test['Survived'] = (final_p > 0.5).astype(int)

    train['Surname'] = train['Name'].str.split(',').str[0]
    test['Surname'] = test['Name'].str.split(',').str[0]
    train['WCG'] = ((train['Sex'] == 'female') | (train['Age'] < 12)).astype(int)
    test['WCG'] = ((test['Sex'] == 'female') | (test['Age'] < 12)).astype(int)

    train_wcg = train[train['WCG'] == 1].groupby('Surname')['Survived'].mean()
    dead_fams = train_wcg[train_wcg == 0].index
    live_fams = train_wcg[train_wcg == 1].index

    test.loc[(test['Surname'].isin(dead_fams)) & (test['WCG'] == 1), 'Survived'] = 0
    test.loc[(test['Surname'].isin(live_fams)) & (test['WCG'] == 1), 'Survived'] = 1

    pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived}).to_csv('submission_god_mode_v2.csv', index=False)
    print("God-Mode V2 Submission Created.")

if __name__ == "__main__":
    main()
