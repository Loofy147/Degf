import pandas as pd
import numpy as np

def main():
    test = pd.read_csv('test.csv')
    # Softmax probabilities ensure they sum to 1.0
    def normalize(row):
        total = sum(row)
        return [x/total for x in row]

    # Simple Length-based G-score heuristic
    def predict(row):
        len_a = len(str(row['response_a']))
        len_b = len(str(row['response_b']))

        if abs(len_a - len_b) < 100:
            return normalize([0.3, 0.3, 0.4])
        elif len_a > len_b:
            return normalize([0.5, 0.2, 0.3])
        else:
            return normalize([0.2, 0.5, 0.3])

    results = test.apply(predict, axis=1, result_type='expand')
    results.columns = ['winner_model_a', 'winner_model_b', 'winner_tie']

    submission = pd.concat([test[['id']], results], axis=1)
    # Ensure ID is first and columns are exact
    submission = submission[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']]
    submission.to_csv('submission_fixed.csv', index=False)
    print("Fixed submission saved.")

if __name__ == "__main__":
    main()
