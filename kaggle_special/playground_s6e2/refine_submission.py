import pandas as pd
import numpy as np
from heart_disease_reasoner import apply_reasoning
from sklearn.metrics import accuracy_score

def main():
    train = pd.read_csv('kaggle_special/playground_s6e2/train.csv')
    test = pd.read_csv('kaggle_special/playground_s6e2/test.csv')
    oof_baseline = np.load('kaggle_special/playground_s6e2/oof_baseline.npy')
    test_baseline_proba = np.load('kaggle_special/playground_s6e2/test_baseline_proba.npy')

    train['Target'] = (train['Heart Disease'] == 'Presence').astype(int)

    print("Applying reasoning to OOF predictions...")
    oof_refined = apply_reasoning(train, oof_baseline)

    baseline_score = accuracy_score(train['Target'], (oof_baseline > 0.5).astype(int))
    refined_score = accuracy_score(train['Target'], (oof_refined > 0.5).astype(int))

    print(f"Baseline OOF Accuracy: {baseline_score:.6f}")
    print(f"Refined OOF Accuracy:  {refined_score:.6f}")
    print(f"Improvement:           {refined_score - baseline_score:.6f}")

    print("\nApplying reasoning to test predictions...")
    test_refined_proba = apply_reasoning(test, test_baseline_proba)

    # Save refined submission
    submission = pd.DataFrame({'id': test.id, 'Heart Disease': np.where(test_refined_proba > 0.5, 'Presence', 'Absence')})
    submission.to_csv('kaggle_special/playground_s6e2/submission_refined.csv', index=False)
    print("Refined submission saved to kaggle_special/playground_s6e2/submission_refined.csv")

if __name__ == "__main__":
    main()
