import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def get_title(name):
    if 'Mrs' in name or 'Mme' in name: return 'Mrs'
    if 'Miss' in name or 'Mlle' in name or 'Ms' in name: return 'Miss'
    if 'Mr' in name: return 'Mr'
    if 'Master' in name: return 'Master'
    return 'Rare'

def process(df):
    df = df.copy()
    df['Title'] = df['Name'].apply(get_title)
    df['Title'] = pd.factorize(df['Title'])[0]
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())

    # Reasoning-driven feature: Fare per Person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    # Reasoning-driven feature: Deck from Cabin
    df['Deck'] = pd.factorize(df['Cabin'].str.get(0).fillna('U'))[0]

    features = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Title', 'FarePerPerson', 'Deck']
    return df[features]

X_train = process(train)
y_train = train['Survived']
X_test = process(test)

m1 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
m2 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)

model = VotingClassifier(estimators=[('rf', m1), ('gb', m2)], voting='soft')
model.fit(X_train, y_train)

preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}).to_csv('submission_ultimate.csv', index=False)
print("Ultimate Titanic submission created.")
