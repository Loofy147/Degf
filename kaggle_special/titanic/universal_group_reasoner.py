import pandas as pd
import numpy as np

# Titanic Top 10% Logic: Surname + Ticket Grouping
# This encodes the genuine reasoning that families/groups survived or died together.

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocessing
train['Surname'] = train['Name'].str.split(',').str[0]
test['Surname'] = test['Name'].str.split(',').str[0]

# Define Woman-Child Groups
train['IsWomanOrChild'] = ((train['Sex'] == 'female') | (train['Age'] < 12)).astype(int)
test['IsWomanOrChild'] = ((test['Sex'] == 'female') | (test['Age'] < 12)).astype(int)

# Group logic by Surname and Ticket
# We look for groups where everyone survives or everyone dies
for df in [train, test]:
    df['FamilyGroup'] = df['Surname'] + df['Ticket'].str[:-1]

# Extract survival patterns from Train
survival_rates = train.groupby('FamilyGroup')['Survived'].mean()
group_counts = train.groupby('FamilyGroup')['Survived'].count()

# Baseline: Gender model
test['Survived'] = (test['Sex'] == 'female').astype(int)

# Reasoning-Driven Overlay
for i, row in test.iterrows():
    if row['IsWomanOrChild']:
        if row['FamilyGroup'] in survival_rates:
            if survival_rates[row['FamilyGroup']] == 0:
                test.loc[i, 'Survived'] = 0
            elif survival_rates[row['FamilyGroup']] == 1:
                test.loc[i, 'Survived'] = 1
    else: # Adult Males
        if row['FamilyGroup'] in survival_rates:
            if survival_rates[row['FamilyGroup']] == 1:
                test.loc[i, 'Survived'] = 1

pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived}).to_csv('submission_universal.csv', index=False)
print("Universal Group Reasoner Created.")
