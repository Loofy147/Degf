import pandas as pd
import numpy as np
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def eng(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]

    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_cols] = df[exp_cols].fillna(0)
    df['TotalSpent'] = df[exp_cols].sum(axis=1)

    # Latent Logic: Families spend similarly and are transported together
    df['GroupSpent'] = df.groupby('Group')['TotalSpent'].transform('mean')
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')

    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side']:
        df[col] = pd.factorize(df[col].fillna('U'))[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())
    features = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Age', 'TotalSpent', 'GroupSpent', 'GroupSize']
    return df[features]

X_train = eng(train)
y_train = train['Transported'].astype(int)
X_test = eng(test)

model = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.01, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test).astype(bool)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_mega.csv', index=False)
print("Spaceship Mega Logic Created.")
