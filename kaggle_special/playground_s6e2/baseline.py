import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

def main():
    train_path = 'kaggle_special/playground_s6e2/train.csv'
    test_path = 'kaggle_special/playground_s6e2/test.csv'

    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train['Target'] = (train['Heart Disease'] == 'Presence').astype(int)

    features = [
        'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
        'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
        'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
    ]

    X = train[features]
    y = train['Target']
    X_test = test[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    print("Training XGBoost baseline with 5-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Using fast parameters for quick baseline
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds_val = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds_val
        test_preds += model.predict_proba(X_test)[:, 1] / 5

        score = accuracy_score(y_val, (preds_val > 0.5).astype(int))
        print(f"  Fold {fold} Accuracy: {score:.5f}")

    total_score = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"Overall OOF Accuracy: {total_score:.5f}")

    # Save probabilities for refinement
    np.save('kaggle_special/playground_s6e2/oof_baseline.npy', oof_preds)
    np.save('kaggle_special/playground_s6e2/test_baseline_proba.npy', test_preds)

    # Save baseline submission
    submission = pd.DataFrame({'id': test.id, 'Heart Disease': np.where(test_preds > 0.5, 'Presence', 'Absence')})
    submission.to_csv('kaggle_special/playground_s6e2/baseline_submission.csv', index=False)
    print("Baseline submission saved to kaggle_special/playground_s6e2/baseline_submission.csv")

if __name__ == "__main__":
    main()
