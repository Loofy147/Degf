import pandas as pd
import numpy as np

def apply_reasoning(df, preds_proba):
    """
    Apply 'Genuine Reasoning' overlay (G-grounding logic) to refine ML predictions.
    Based on high-certainty rules discovered in the training set:
    - Thallium=7 & Vessels>=2 (96.6% Presence)
    - Thallium=7 & ST depression > 2.0 (96% Presence)
    - Thallium=3 & Vessels=0 & ST depression=0 (91.2% Absence)

    This is the final refined version, using a low G-weight that provided a small gain.
    """
    refined_preds = preds_proba.copy()

    # Weight of reasoning (G-weight)
    g_weight = 0.05

    # Presence rules
    rule1 = (df['Thallium'] == 7) & (df['Number of vessels fluro'] >= 2)
    rule2 = (df['Thallium'] == 7) & (df['ST depression'] > 2.0)

    # Absence rules
    rule3 = (df['Thallium'] == 3) & (df['Number of vessels fluro'] == 0) & (df['ST depression'] == 0)

    # Combined prediction (G-grounded synthesis)
    refined_preds[rule1] = (1 - g_weight) * refined_preds[rule1] + g_weight * 0.966
    refined_preds[rule2] = (1 - g_weight) * refined_preds[rule2] + g_weight * 0.960
    refined_preds[rule3] = (1 - g_weight) * refined_preds[rule3] + g_weight * 0.088

    return refined_preds

if __name__ == "__main__":
    print("Logical reasoner logic finalized with low G-weight.")
