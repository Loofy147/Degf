import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    df['HomePlanet'] = pd.factorize(df['HomePlanet'].fillna('Earth'))[0]
    df['CryoSleep'] = pd.factorize(df['CryoSleep'].fillna(False))[0]
    df['Deck'] = pd.factorize(df['Cabin'].str.split('/', expand=True)[0].fillna('U'))[0]
    df['Side'] = pd.factorize(df['Cabin'].str.split('/', expand=True)[2].fillna('U'))[0]
    df['Age'] = df['Age'].fillna(df['Age'].median())

    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_cols] = df[exp_cols].fillna(0)
    df['Spent'] = df[exp_cols].sum(axis=1)

    # Reasoning-driven: Log transform spending
    df['SpentLog'] = np.log1p(df['Spent'])

    features = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Age', 'SpentLog']
    return df[features]

X_train = process(train)
y_train = train['Transported']
X_test = process(test)

m1 = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
m2 = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

model = VotingClassifier(estimators=[('rf', m1), ('gb', m2)], voting='soft')
model.fit(X_train, y_train)

preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_ultimate.csv', index=False)
print("Ultimate Spaceship Titanic submission created.")
