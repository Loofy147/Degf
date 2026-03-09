import pandas as pd
import numpy as np
from heart_disease_reasoner import apply_reasoning

def test_apply_reasoning():
    # Create a small dummy dataframe
    data = {
        'Thallium': [7, 7, 3, 3, 5],
        'Number of vessels fluro': [2, 0, 0, 1, 0],
        'ST depression': [0.0, 2.5, 0.0, 0.0, 1.0]
    }
    df = pd.DataFrame(data)
    preds_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    refined = apply_reasoning(df, preds_proba)

    # Assertions based on rules in heart_disease_reasoner.py
    # Rule 1: Thallium=7 & Vessels>=2 -> Boost
    assert refined[0] > 0.5, f"Expected boost for Rule 1, got {refined[0]}"

    # Rule 2: Thallium=7 & ST depression > 2.0 -> Boost
    assert refined[1] > 0.5, f"Expected boost for Rule 2, got {refined[1]}"

    # Rule 3: Thallium=3 & Vessels=0 & ST depression=0 -> Penalize
    assert refined[2] < 0.5, f"Expected penalty for Rule 3, got {refined[2]}"

    # No rule active
    assert refined[3] == 0.5, f"Expected no change for index 3, got {refined[3]}"
    assert refined[4] == 0.5, f"Expected no change for index 4, got {refined[4]}"

    print("Heart Disease Reasoner unit tests passed.")

if __name__ == "__main__":
    test_apply_reasoning()
