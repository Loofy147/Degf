import pandas as pd
import numpy as np
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def get_wcg(df):
    df['Surname'] = df['Name'].str.split(',').str[0]
    # WCG rule
    df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Age'] < 12) | (df['Name'].str.contains('Master'))).astype(int)
    return df

train = get_wcg(train)
test = get_wcg(test)

# Base model: XGBoost on simple features
def eng(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Age'] = df['Age'].fillna(28)
    df['Fare'] = df['Fare'].fillna(14.45)
    df['Embarked'] = pd.factorize(df['Embarked'].fillna('S'))[0]
    return df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(eng(train), train['Survived'])

# Base Preds
test['Survived'] = model.predict(eng(test))

# WCG Logic Refinement (Genuine reasoning overlay)
# Groups of Women/Children that shared a ticket or surname
train_wcg = train[train['IsWomanOrChild'] == 1].groupby('Surname')['Survived'].mean()
dead_families = train_wcg[train_wcg == 0].index
live_families = train_wcg[train_wcg == 1].index

test.loc[(test['Surname'].isin(dead_families)) & (test['IsWomanOrChild'] == 1), 'Survived'] = 0
test.loc[(test['Surname'].isin(live_families)) & (test['IsWomanOrChild'] == 1), 'Survived'] = 1

pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived}).to_csv('submission_ultimate_refined.csv', index=False)
print("Ultimate Refined Titanic Created.")
