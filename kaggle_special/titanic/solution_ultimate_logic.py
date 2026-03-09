import pandas as pd
import numpy as np

# 'Genuine Reasoning' - The 0.81+ logic for Titanic
# Based on the "Titanic: Machine Learning from Disaster" top kernels

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def get_wcg(df):
    df['Surname'] = df['Name'].str.split(',').str[0]
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x[:3])
    # A person is in a "woman-child group" if they are female or a master
    df['IsWomanOrChild'] = ((df['Sex'] == 'female') | (df['Name'].str.contains('Master'))).astype(int)
    return df

train = get_wcg(train)
test = get_wcg(test)

# Group logic: If any woman/child in the family group dies, all in that group are predicted to die
# If any woman/child in the family group survives, all are predicted to survive

test['Survived'] = (test['Sex'] == 'female').astype(int)

# Refined grouping
for surname, group in test.groupby('Surname'):
    if len(group) > 1:
        # Check if this surname exists in train
        train_group = train[train['Surname'] == surname]
        if not train_group.empty:
            # If anyone in the train group survived, follow them
            surv_rate = train_group['Survived'].mean()
            if surv_rate == 1:
                test.loc[test['Surname'] == surname, 'Survived'] = 1
            elif surv_rate == 0:
                test.loc[test['Surname'] == surname, 'Survived'] = 0

# Final check for 3rd class females
test.loc[(test['Sex'] == 'female') & (test['Pclass'] == 3) & (test['Fare'] > 20), 'Survived'] = 0

pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived}).to_csv('submission_ultimate_logic.csv', index=False)
print("Ultimate Logic Submission Created.")
