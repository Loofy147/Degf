import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

# Top-tier logic: Combining WCG with CatBoost and group-based pseudo-labeling
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def eng(df):
    df = df.copy()
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

    # Family Groups based on Surname and Ticket
    df['Surname'] = df['Name'].str.split(',').str[0]
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x[:3])

    # Encode categorical for CatBoost (keep as strings)
    df['Sex'] = df['Sex'].astype('category')
    df['Embarked'] = df['Embarked'].fillna('S').astype('category')
    df['Title'] = df['Title'].astype('category')

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    return df[features], df['Surname']

X_train, S_train = eng(train)
y_train = train['Survived']
X_test, S_test = eng(test)

# CatBoost handles categories natively
model = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.01, cat_features=['Sex', 'Embarked', 'Title'], verbose=0, random_state=42)
model.fit(X_train, y_train)

# Base Preds
test_preds = model.predict(X_test)

# WCG Logic Overlay (The "Genuine reasoning" boost)
test['Survived'] = test_preds
train['Surname'] = S_train
test['Surname'] = S_test
train['IsWomanOrChild'] = ((train['Sex'] == 'female') | (train['Age'] < 12)).astype(int)
test['IsWomanOrChild'] = ((test['Sex'] == 'female') | (test['Age'] < 12)).astype(int)

# If a family in train all died, mark test members as died.
dead_families = train.groupby('Surname')['Survived'].mean()
dead_families = dead_families[dead_families == 0].index
test.loc[(test['Surname'].isin(dead_families)) & (test['IsWomanOrChild'] == 1), 'Survived'] = 0

# If a family in train all survived, mark test members as survived.
live_families = train.groupby('Surname')['Survived'].mean()
live_families = live_families[live_families == 1].index
test.loc[(test['Surname'].isin(live_families)) & (test['IsWomanOrChild'] == 1), 'Survived'] = 1

pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived}).to_csv('submission_god_mode.csv', index=False)
print("Titanic God-Mode Submission Created.")
