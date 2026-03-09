import pandas as pd
import numpy as np
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def tech_eng(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]

    # Impute categorical by Group (Technical consistency)
    for col in ['HomePlanet', 'Destination']:
        df[col] = df.groupby('Group')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'U'))

    # Expenditure vs CryoSleep Logic
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for c in exp_cols: df[c] = df[c].fillna(0)
    df['TotalSpent'] = df[exp_cols].sum(axis=1)

    # If they spend, they can't be in CryoSleep
    df.loc[(df['TotalSpent'] > 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)

    # Cabin Decomposition
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = pd.factorize(df['Deck'].fillna('U'))[0]
    df['Side'] = pd.factorize(df['Side'].fillna('U'))[0]

    for col in ['HomePlanet', 'Destination']:
        df[col] = pd.factorize(df[col])[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())

    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Age', 'TotalSpent']
    return df[features]

X_train = tech_eng(train)
y_train = train['Transported'].astype(int)
X_test = tech_eng(test)

model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test).astype(bool)
pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_tech.csv', index=False)
print("Technical Reasoner Created.")
