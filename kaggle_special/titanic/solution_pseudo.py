import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def eng(df):
    df = df.copy()
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
    df['Title'] = pd.factorize(df['Title'])[0]
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df['Fare'].fillna(train['Fare'].median())
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Family']
    return df[features]

X_train = eng(train)
y_train = train['Survived']
X_test = eng(test)

# Stage 1: Initial Model
m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
m.fit(X_train, y_train)

# Stage 2: Pseudo-Labeling (Only high confidence)
probs = m.predict_proba(X_test)
pseudo_idx = np.where((probs[:, 0] > 0.9) | (probs[:, 1] > 0.9))[0]
X_pseudo = X_test.iloc[pseudo_idx]
y_pseudo = m.predict(X_pseudo)

# Combine
X_combined = pd.concat([X_train, X_pseudo])
y_combined = pd.concat([y_train, pd.Series(y_pseudo)])

# Stage 3: Final Model
model = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01, random_state=42))
], voting='soft')
model.fit(X_combined, y_combined)

preds = model.predict(X_test)
pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}).to_csv('submission_pseudo.csv', index=False)
print("Titanic Pseudo-Labeled Submission Created.")
