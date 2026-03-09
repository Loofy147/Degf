import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def feature_eng(df):
    df = df.copy()
    # Continuous interactions
    df['HR_Vessels'] = df['Max HR'] / (df['Number of vessels fluro'] + 1)
    df['Chest_Thal'] = df['Chest pain type'] * df['Thallium']

    # Categorical interaction hashing (more memory efficient than strings)
    df['Thal_Vess_Chest'] = df['Thallium'] * 100 + df['Number of vessels fluro'] * 10 + df['Chest pain type']
    df['ST_Slope_Sex'] = (df['ST depression'] * 10).astype(int) * 100 + df['Slope of ST'] * 10 + df['Sex']

    return df

def main():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train['Target'] = (train['Heart Disease'] == 'Presence').astype(int)

    X = feature_eng(train)
    X_test = feature_eng(test)

    features = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
                'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium',
                'HR_Vessels', 'Chest_Thal', 'Thal_Vess_Chest', 'ST_Slope_Sex']

    X = X[features]
    y = train['Target']
    X_test = X_test[features]

    print("Training XGBoost V7...")
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    print("Generating submission...")
    preds = model.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({'id': test.id, 'Heart Disease': (preds > 0.5).astype(int)})
    sub.to_csv('submission_v7.csv', index=False)
    print("V7 Submission Saved.")

if __name__ == "__main__":
    main()
