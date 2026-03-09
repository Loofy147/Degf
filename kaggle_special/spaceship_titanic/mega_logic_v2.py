import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def feature_eng(df):
    df = df.copy()
    df['GroupId'] = df['PassengerId'].str.split('_', expand=True)[0]

    # Within each group, HomePlanet and Destination are almost always identical
    for col in ['HomePlanet', 'Destination']:
        df[col] = df.groupby('GroupId')[col].ffill().bfill()

    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[exp_cols] = df[exp_cols].fillna(0)
    df['Spent'] = df[exp_cols].sum(axis=1)
    df.loc[(df['Spent'] > 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False
    df.loc[(df['Spent'] == 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = True

    for col in ['HomePlanet', 'CryoSleep', 'Destination']:
        df[col] = pd.factorize(df[col].fillna('U'))[0]

    cabin = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = pd.factorize(cabin[0].fillna('U'))[0]
    df['Side'] = pd.factorize(cabin[2].fillna('U'))[0]

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')

    return df[['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'Age', 'Spent', 'GroupSize']]

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X = feature_eng(train)
    y = train['Transported'].astype(int)
    X_test = feature_eng(test)

    m1 = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=42)
    m1.fit(X, y)
    p1 = m1.predict_proba(X_test)[:, 1]

    m2 = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.01, verbose=0, random_state=42)
    m2.fit(X, y)
    p2 = m2.predict_proba(X_test)[:, 1]

    final_p = (p1 + p2) / 2
    preds = (final_p > 0.5).astype(bool)

    pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': preds}).to_csv('submission_mega_v2.csv', index=False)
    print("Mega-Logic V2 Submission Created.")

if __name__ == "__main__":
    main()
