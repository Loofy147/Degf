import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def preprocess(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_cols] = df[exp_cols].fillna(0)
    df['TotalSpending'] = df[exp_cols].sum(axis=1)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)

    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']:
        df[col] = pd.factorize(df[col].fillna('Missing'))[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())
    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'TotalSpending', 'NoSpending', 'Deck', 'Side']
    return df[features]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = preprocess(train)
y_train = train['Transported'].astype(int)
X_test = preprocess(test)

clf1 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf2 = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42)
clf3 = LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42, verbosity=-1)

eclf = VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('lgbm', clf3)], voting='soft')
eclf.fit(X_train, y_train)

preds = eclf.predict(X_test).astype(bool)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_ensemble.csv', index=False)
print("Spaceship Titanic ensemble submission created.")
