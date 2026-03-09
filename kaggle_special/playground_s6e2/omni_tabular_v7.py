import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

def feature_eng(df):
    df = df.copy()

    # 1. High-Order Interactions (The "Reasoning Clusters")
    df['Thal_Vess_Chest'] = (df['Thallium'].astype(str) + "_" +
                             df['Number of vessels fluro'].astype(str) + "_" +
                             df['Chest pain type'].astype(str))

    df['ST_Slope_Sex'] = (df['ST depression'].astype(str) + "_" +
                          df['Slope of ST'].astype(str) + "_" +
                          df['Sex'].astype(str))

    # 2. Medical Ratios (Biological grounding)
    df['HR_Pressure_Interaction'] = df['Max HR'] / (df['BP'] + 1)
    df['Age_ST_Product'] = df['Age'] * df['ST depression']

    # 3. Categorical Treatment
    cat_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
                    'Exercise angina', 'Slope of ST', 'Thallium',
                    'Thal_Vess_Chest', 'ST_Slope_Sex']

    for col in cat_features:
        df[col] = df[col].astype(str)

    return df, cat_features

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train['Target'] = (train['Heart Disease'] == 'Presence').astype(int)

    train_eng, cat_features = feature_eng(train)
    test_eng, _ = feature_eng(test)

    features = [c for c in train_eng.columns if c not in ['id', 'Heart Disease', 'Target']]
    X = train_eng[features]
    y = train['Target']
    X_test = test_eng[features]

    # 10-Fold for maximum alignment
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    print(f"Training Elite Omni-Tabular V7 (10-Fold CatBoost)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Heavy model with tuned parameters for 0.95+
        model = CatBoostClassifier(
            iterations=5000,
            depth=7,
            learning_rate=0.02,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_strength=1,
            cat_features=cat_features,
            verbose=0,
            random_state=fold,
            task_type='CPU' # Use GPU if available in real env
        )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 10

        score = accuracy_score(y_val, (oof_preds[val_idx] > 0.5).astype(int))
        print(f"  Fold {fold} Accuracy: {score:.6f} (Iter: {model.best_iteration_})")

    final_score = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"Omni-Tabular V7 OOF Accuracy: {final_score:.6f}")

    np.save('oof_v7.npy', oof_preds)
    np.save('test_v7_proba.npy', test_preds)

    sub = pd.DataFrame({'id': test.id, 'Heart Disease': (test_preds > 0.5).astype(int)})
    sub.to_csv('submission_v7.csv', index=False)
    print("Omni-Tabular V7 submission saved.")

if __name__ == "__main__":
    main()
