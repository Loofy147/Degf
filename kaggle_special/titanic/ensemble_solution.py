import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def preprocess(df, train_df=None):
    df = df.copy()
    # Feature Engineering
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
    df['Title'] = pd.factorize(df['Title'])[0]

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Embarked'] = pd.factorize(df['Embarked'].fillna('S'))[0]
    df['Fare'] = df['Fare'].fillna(train_df['Fare'].median() if train_df is not None else df['Fare'].median())

    # Advanced: Ticket frequency
    ticket_counts = df['Ticket'].value_counts().to_dict()
    df['TicketFreq'] = df['Ticket'].map(ticket_counts)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'TicketFreq']
    return df[features]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = preprocess(train)
y_train = train['Survived']
X_test = preprocess(test, train)

# Ensemble
clf1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf2 = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
clf3 = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1)
clf4 = CatBoostClassifier(n_estimators=100, depth=5, verbose=0, random_state=42)

eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('lgbm', clf3), ('cat', clf4)], voting='soft')
eclf.fit(X_train, y_train)

preds = eclf.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}).to_csv('submission_ensemble.csv', index=False)
print("Titanic ensemble submission created.")
