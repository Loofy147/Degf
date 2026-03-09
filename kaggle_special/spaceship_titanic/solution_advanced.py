import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def preprocess(df):
    df = df.copy()

    # Feature Engineering based on Top 10% kernels
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df['Solo'] = (df['GroupSize'] == 1).astype(int)

    # Log transform expenditure
    exp_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in exp_features:
        df[col] = df[col].fillna(0)
    df['TotalSpent'] = df[exp_features].sum(axis=1)
    df['HasSpent'] = (df['TotalSpent'] > 0).astype(int)

    # Impute categorical
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = pd.factorize(df[col])[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())

    features = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Age', 'VIP', 'TotalSpent', 'HasSpent', 'GroupSize', 'Solo']
    return df[features]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = preprocess(train)
y_train = train['Transported'].astype(int)
X_test = preprocess(test)

# Advanced Ensemble
clf1 = XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=42)
clf2 = LGBMClassifier(n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
clf3 = CatBoostClassifier(n_estimators=1000, depth=6, learning_rate=0.01, rsm=0.8, verbose=0, random_state=42)

model = VotingClassifier(estimators=[('xgb', clf1), ('lgbm', clf2), ('cat', clf3)], voting='soft')
model.fit(X_train, y_train)

preds = model.predict(X_test).astype(bool)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_advanced.csv', index=False)
print("Spaceship Advanced Submission Created.")
