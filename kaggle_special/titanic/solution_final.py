import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['Family'] == 1).astype(int)
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())
    df['Embarked'] = pd.factorize(df['Embarked'].fillna('S'))[0]

    # Critical: Use only the most robust features to reach top scores
    features = ['Pclass', 'Sex', 'Age', 'Family', 'Fare', 'Title', 'Embarked']
    return df[features]

X_train = process(train)
y_train = train['Survived']
X_test = process(test)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}).to_csv('submission_final.csv', index=False)
print("Final Titanic submission created.")
