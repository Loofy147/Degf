import pandas as pd
import numpy as np

# 'Genuine Reasoning' logic for Spaceship Titanic
# If a group (PassengerId first 4 digits) has people transported, others likely are too.

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def get_logic(df):
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
    return df

train = get_logic(train)
test = get_logic(test)

# Base: CryoSleep is the strongest predictor
test['Transported'] = test['CryoSleep'].fillna(False)

# Group logic: If someone in the same group in train was transported
group_transported = train.groupby('Group')['Transported'].mean()
test['Group_Surv'] = test['Group'].map(group_transported)

test.loc[test['Group_Surv'] == 1, 'Transported'] = True
test.loc[test['Group_Surv'] == 0, 'Transported'] = False

pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': test.Transported}).to_csv('submission_group.csv', index=False)
print("Spaceship Group Logic Submission Created.")
