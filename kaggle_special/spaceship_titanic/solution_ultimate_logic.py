import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

    # Expenditure logic: People in CryoSleep spend 0
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for c in exp_cols:
        df[c] = df[c].fillna(0)
    df['TotalSpent'] = df[exp_cols].sum(axis=1)

    # Impute VIP and CryoSleep based on spending
    df.loc[(df['TotalSpent'] > 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False
    df.loc[(df['TotalSpent'] == 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = True

    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']:
        df[col] = pd.factorize(df[col].fillna('U'))[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())

    return df[['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Age', 'VIP', 'TotalSpent', 'GroupSize']]

X_train = process(train)
y_train = train['Transported'].astype(int)
X_test = process(test)

# RandomForest with deeper trees to capture the reasoning paths
model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test).astype(bool)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_ultimate_logic.csv', index=False)
print("Ultimate Spaceship Logic Created.")
