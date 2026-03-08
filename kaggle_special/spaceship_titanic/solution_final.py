import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    df['HomePlanet'] = pd.factorize(df['HomePlanet'].fillna('Earth'))[0]
    df['CryoSleep'] = pd.factorize(df['CryoSleep'].fillna(False))[0]
    df['Deck'] = pd.factorize(df['Cabin'].str.split('/', expand=True)[0].fillna('U'))[0]
    df['Side'] = pd.factorize(df['Cabin'].str.split('/', expand=True)[2].fillna('U'))[0]
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['VIP'] = pd.factorize(df['VIP'].fillna(False))[0]

    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_cols] = df[exp_cols].fillna(0)
    df['Spent'] = df[exp_cols].sum(axis=1)

    features = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Age', 'VIP', 'Spent']
    return df[features]

X_train = process(train)
y_train = train['Transported']
X_test = process(test)

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_final.csv', index=False)
print("Final Spaceship Titanic submission created.")
