# Kaggle Competitions with DEGF

This directory contains implementations for Kaggle competitions using models enhanced by the Dynamic Entropy Genuineness Framework (DEGF).

## Directory Structure
- `titanic/`: Titanic - Machine Learning from Disaster
- `spaceship_titanic/`: Spaceship Titanic
- `house_prices/`: House Prices - Advanced Regression Techniques

## Performance Reflection (DEGF-DeepSeek)
Through extensive "Thermodynamic Grounding" and iterative logic refinement, we established high-performing reasoning-based baselines.

### Key Reasoning Patterns Identified
1. **Woman-Child Group (WCG) Heuristic**: A powerful logical anchor for the Titanic dataset, confirmed by Q2 head activity.
2. **Technical Ship Consistency**: Spaceship Titanic groups exhibit technical dependencies (HomePlanet, Destination) that can be used for robust imputation.
3. **Domain Feature Interaction**: For House Prices, the interaction between Quality and Age provides a higher-G reasoning signal than raw features.

### Best Verified Scores
- **Titanic**: 0.80622 (WCG Logic) - Significant improvement over baseline.
- **Spaceship Titanic**: 0.74140 (Baseline)
- **House Prices**: 0.13909 RMSE (Baseline)

## Workflow
1. **G-Grounding**: Use `train_grounded_titanic_v2.py` style scripts to align Q2 heads with dataset-specific logic.
2. **SGS-2 Inference**: Use `sgs2_prototype_fast.py` for deliberative code generation.
3. **Logic Overlay**: Combine standard ML models (XGBoost/CatBoost) with explicit "Genuine Reasoning" rules for maximum accuracy.

## Leaderboard Strategy
The path to the Top 10 relies on **Group-Based Logic** and **Pseudo-Labeling** where latent model states (G-scores) guide the confidence of the assigned labels.
