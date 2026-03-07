# Kaggle Competitions with DEGF

This directory contains implementations for Kaggle competitions using models enhanced by the Dynamic Entropy Genuineness Framework (DEGF).

## Directory Structure
- `titanic/`: Titanic - Machine Learning from Disaster
- `spaceship_titanic/`: Spaceship Titanic
- `house_prices/`: House Prices - Advanced Regression Techniques

## Workflow
1. Use `train_deepseek_v6.py` to generate thermodynamic delta weights.
2. Apply weights to the base model using the DEGF logic.
3. Generate or refine solution scripts using the enhanced model's reasoning capabilities.
4. Run `solution.py` in each directory to generate `submission.csv`.
5. Submit via Kaggle API.

## Baseline Scores (DEGF-DeepSeek-Baseline)
- Titanic: 0.78229
- Spaceship Titanic: 0.74140
- House Prices: 0.13909 (RMSE)
