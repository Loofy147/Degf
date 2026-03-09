import pandas as pd
import numpy as np

def compute_g_score_simple(text):
    length = len(str(text))
    if length < 50: return 0.2
    if "As an AI" in str(text): return 0.1
    if "complex issue" in str(text): return 0.4

    genuine_markers = ["However", "On the other hand", "Therefore", "Specifically", "Furthermore"]
    marker_count = sum(1 for m in genuine_markers if m in str(text))

    return np.clip(0.3 + 0.1 * marker_count + (length / 5000), 0, 1)

def predict_winner(row):
    g_a = compute_g_score_simple(row['response_a'])
    g_b = compute_g_score_simple(row['response_b'])

    # Probabilistic preference based on G-scores
    diff = g_a - g_b
    if abs(diff) < 0.05:
        return [0.2, 0.2, 0.6] # Prefer Tie
    elif diff > 0.05:
        # A is more genuine
        return [0.7, 0.1, 0.2]
    else:
        # B is more genuine
        return [0.1, 0.7, 0.2]

def main():
    test = pd.read_csv('kaggle_special/llm_finetuning/test.csv')

    print("Classifying response genuineness using DEGF heuristics (Probabilistic)...")
    results = []
    for _, row in test.iterrows():
        results.append(predict_winner(row))

    preds = pd.DataFrame(results, columns=['winner_model_a', 'winner_model_b', 'winner_tie'])
    submission = pd.concat([test[['id']], preds], axis=1)
    submission.to_csv('kaggle_special/llm_finetuning/submission_degf.csv', index=False)
    print("DEGF heuristic submission saved.")

if __name__ == "__main__":
    main()
