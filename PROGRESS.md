# Progress

Random seed: 42 (fixed across all notebooks)

## Setup

- [x] Virtual environment created, packages installed
- [x] AIF360 verified working — Reweighing, ClassificationMetric, EqOddsPostprocessing tested end to end
- [x] Jupyter and Data Wrangler extensions installed, ipykernel pinned below 7
- [x] Repository structure created
- [x] Dissertation manuscript gitignored
- [x] Dataset placed at data/raw/ibm_dataset.csv

## Data preparation

- [x] Shape verified at 1,470 × 35
- [x] Dtypes checked, no missing values, no placeholder-encoded missingness
- [x] No duplicate rows
- [x] Target balance recorded: 1,233 retained / 237 left, 83.88% / 16.12%
- [x] Constant and identifier columns found by property
- [x] Dropped EmployeeCount, Over18, StandardHours, EmployeeNumber
- [x] cleaned_ibm_dataset.csv saved at 1,470 × 34
- [x] gender_priv, age_priv, marital_priv added
- [x] Group gaps recorded: age 18.99/10.92, marital 19.20/12.48, gender 14.80/17.01
- [x] Categorical columns one-hot encoded, 7 into 21
- [x] Numeric columns standardised, fitted on training data only
- [x] Stratified 80/20 split
- [x] train_dataset.csv saved at 1,176 × 48
- [x] test_dataset.csv saved at 294 × 48
- [x] Stratification verified at 16.16% / 15.99%, no overlapping rows
- [x] preprocessor.joblib saved
- [x] Notebook saved to disk, 13 cells
- [ ] Remove the "Chapter 3.2" reference from the loading markdown    <-- current
- [ ] Change select_dtypes(include="object") to select_dtypes(exclude="number")
- [ ] First git commit

## Model building

- [ ] Load train and test datasets, separate features, target and protected columns
- [ ] Compute class weights: 0.596 and 3.095
- [ ] Define Logistic Regression with class weighting
- [ ] Define Decision Tree with class weighting
- [ ] Define Random Forest with class weighting
- [ ] Define XGBoost with scale_pos_weight
- [ ] 10-fold stratified cross-validation on all four
- [ ] Fit final models on full training set
- [ ] Predict on held-out test set
- [ ] Write shared measurement suite, called identically by all three rounds
- [ ] Bootstrap 95% confidence intervals
- [ ] Performance metrics: accuracy, precision, recall, F1, AUC-ROC, balanced accuracy, PR-AUC, Brier
- [ ] Fairness metrics: SPD, AOD, PPD across gender, age, marital status
- [ ] Save models, predictions and baseline results table

## Reweighing

- [ ] Apply AIF360 Reweighing to training data only
- [ ] Inspect resulting instance weights
- [ ] Retrain all four models on reweighted data
- [ ] Re-run the identical measurement suite
- [ ] Save models, predictions and results table

## Equalised odds

- [ ] Fit AIF360 EqOddsPostprocessing on held-out validation predictions
- [ ] Apply threshold adjustment to test predictions for all four models
- [ ] Re-run the identical measurement suite
- [ ] Save predictions and results table

## Explainability

- [ ] SHAP global importance, pre-mitigation, all four models
- [ ] SHAP global importance, post-mitigation
- [ ] Local SHAP explanations for privileged and unprivileged cases
- [ ] Compare SHAP importance of protected attributes against measured SPD, AOD, PPD

## Results

- [ ] Baseline vs reweighing vs equalised odds comparison table
- [ ] Fairness change against performance change
- [ ] Trade-off figures
- [ ] Export all tables and figures
- [ ] Write the Findings and Results section

## Repository

- [ ] README
- [ ] Clean end-to-end re-run from restarted kernel
