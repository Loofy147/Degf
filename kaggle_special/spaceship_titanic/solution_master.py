import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]

    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for c in exp_cols:
        df[c] = df[c].fillna(0)
    df['TotalSpent'] = df[exp_cols].sum(axis=1)

    # Logic: If they spend money, they are NOT in CryoSleep
    df.loc[(df['TotalSpent'] > 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False
    # Logic: If they spend 0 and are under 13, they are likely transported
    df['IsChild'] = (df['Age'] < 13).astype(int)

    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']:
        df[col] = pd.factorize(df[col].fillna('U'))[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())

    cols = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Age', 'TotalSpent', 'IsChild']
    return df[cols]

X_train = process(train)
y_train = train['Transported'].astype(int)
X_test = process(test)

model = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test).astype(bool)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_master.csv', index=False)
print("Spaceship Master Submission Created.")
