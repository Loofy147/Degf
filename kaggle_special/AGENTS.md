# Kaggle Competitions with DEGF

This directory contains implementations for Kaggle competitions using models enhanced by the Dynamic Entropy Genuineness Framework (DEGF).

## Directory Structure
- `titanic/`: Titanic - Machine Learning from Disaster
- `spaceship_titanic/`: Spaceship Titanic
- `house_prices/`: House Prices - Advanced Regression Techniques

## Performance Reflection (DEGF-DeepSeek)
Through iterative refinement and bulk submission, we observed that the **Baseline DEGF Model** weights often outperformed over-engineered variants. This supports the framework's hypothesis that genuine reasoning (Q2 heads) is more effective when allowed to operate on raw features rather than being constrained by human-engineered complexity that might introduce noise.

### Best Verified Scores
- **Titanic**: 0.78229 (Baseline)
- **Spaceship Titanic**: 0.74140 (Baseline)
- **House Prices**: 0.13909 RMSE (Baseline)

## Workflow
1. Use `train_deepseek_v6_enhanced.py` to generate thermodynamic delta weights.
2. Apply weights to the base model using the DEGF logic.
3. Run `solution.py` in each directory for the most robust results.
4. For deeper reasoning, use `sgs2_prototype_fast.py` to generate deliberative logic paths.

## Leaderboard Strategy
To reach the Top 10, future work should focus on **Pseudo-Labeling** and **Target Encoding** driven by high-G latent states, as suggested by the model's self-critique.
