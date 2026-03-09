import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def get_clusters(df):
    df['Surname'] = df['Name'].str.split(',').str[0]
    df['TicketPrefix'] = df['Ticket'].str[:-2] # Group by similar tickets
    df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Age'] < 12)).astype(int)
    return df

train = get_clusters(train)
test = get_clusters(test)

# Logic: Family/Ticket groups are the best "reasoning units"
# If a WomanOrChild in a group died, others likely died.
# If a Male in a group lived, others likely lived.

test['Survived'] = (test['Sex'] == 'female').astype(int)

# Extract survival rate per group
train_groups = train.groupby(['Surname', 'Pclass'])['Survived'].mean()

for i, row in test.iterrows():
    group_key = (row['Surname'], row['Pclass'])
    if group_key in train_groups:
        rate = train_groups[group_key]
        if row['IsWomanOrChild']:
            if rate == 0: test.loc[i, 'Survived'] = 0
        else:
            if rate == 1: test.loc[i, 'Survived'] = 1

pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived}).to_csv('submission_cluster.csv', index=False)
print("Logical Cluster Reasoner Created.")
