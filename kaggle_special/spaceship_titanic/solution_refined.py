import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def refine_spaceship(df):
    df = df.copy()
    # Cabin decomposition
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')

    # Expenditure
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_cols] = df[exp_cols].fillna(0)
    df['TotalSpending'] = df[exp_cols].sum(axis=1)

    # Categorical
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']:
        df[col] = df[col].fillna('Missing')
        df[col] = pd.factorize(df[col])[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())

    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'TotalSpending', 'Deck', 'Side']
    return df[features]

X_train = refine_spaceship(train)
y_train = train['Transported']
X_test = refine_spaceship(test)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds})
output.to_csv('submission_refined.csv', index=False)
print("Refined Spaceship Titanic submission created.")
