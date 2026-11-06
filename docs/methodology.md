# Methodology

This document covers the full method behind the [project README](../README.md): how the data was prepared, how the models were built and measured, how fairness was defined, and the limits of the conclusions.

**Contents**

1. [Data preparation](#1-data-preparation)
2. [Protected attributes](#2-protected-attributes)
3. [Models](#3-models)
4. [Measurement suite](#4-measurement-suite)
5. [Bias mitigation](#5-bias-mitigation)
6. [Explainability](#6-explainability)
7. [Reproducibility](#7-reproducibility)
8. [Limitations](#8-limitations)

---

## 1. Data preparation

Notebook: [`01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb)

| Step | Detail |
|---|---|
| Audit | 1,470 rows × 35 columns. No missing values and no duplicate rows. Object columns were also checked for missingness hidden as placeholder strings ("unknown", "n/a", "?"). |
| Target balance | 1,233 retained (83.88%) and 237 left (16.12%), an imbalance of 5.2 : 1. |
| Column removal | Found by property rather than by name. `EmployeeCount`, `Over18` and `StandardHours` are constant, and `EmployeeNumber` is unique per row. |
| Split | Stratified 80/20 split on attrition: 1,176 training and 294 test rows. The attrition rate is 16.16% in train and 15.99% in test, with no rows shared between them. |
| Encoding | 23 numeric columns standardised. 7 categorical columns one-hot encoded with the first level dropped, giving 21 columns and **44 features** in total. |
| Leakage control | The `ColumnTransformer` is fitted on the training set only and then applied to the test set. It is saved as `preprocessor.joblib`. |

The split is created **once** and written to disk. Every later stage reads those same files, so all three measurement rounds use identical rows.

## 2. Protected attributes

Each attribute is binarised into a privileged group (1) and an unprivileged group (0), following AI Fairness 360 convention. The protected columns are held out of the feature matrix as audit labels. The underlying `Age`, `Gender` and `MaritalStatus` features stay in the model inputs.

| Attribute | Privileged (1) | Unprivileged (0) | n (unpriv / priv) | Attrition rate (unpriv / priv) |
|---|---|---|---|---|
| Age | 40 and over | Under 40 | 948 / 522 | 18.99% / 10.92% |
| Marital status | Married | Single or divorced | 797 / 673 | 19.20% / 12.48% |
| Gender | Male | Female | 588 / 882 | 14.80% / 17.01% |

Age and marital status already show clear gaps in the raw data. Gender is the narrowest gap, and the only one running the other way.

## 3. Models

Notebook: [`02_model_building.ipynb`](../notebooks/02_model_building.ipynb)

The four classifiers were chosen to run from a directly readable model to opaque ensembles. That gradient lets the explainability question be tested across it.

| Model | Configuration |
|---|---|
| Logistic Regression | `class_weight` balanced (0.596 / 3.095), `max_iter=1000` |
| Decision Tree | `class_weight` balanced |
| Random Forest | 300 trees, `class_weight` balanced |
| XGBoost | `scale_pos_weight` = 986 / 190 ≈ 5.19 |

All models use `random_state=42`. Stability was checked with 10-fold stratified cross-validation on the training set. Each model was then refitted on the full training set and scored once on the held-out test set.

## 4. Measurement suite

Module: [`src/metrics.py`](../src/metrics.py)

One function, `measure_all`, scores every round: baseline, reweighing and equalised odds. The code never changes between rounds, so any difference in the results comes from the mitigation and not from the measurement.

### Performance (positive class = attrition)

Accuracy, precision, recall, F1, ROC-AUC, **balanced accuracy**, **PR-AUC** and **Brier score**. With 16% attrition, accuracy rewards a model for predicting "stays". The headline metrics are therefore the four in bold, which are robust to imbalance or measure calibration.

### Fairness (favourable outcome = retention)

Each fairness metric is computed as **unprivileged minus privileged**. Zero means parity, and a negative value means the unprivileged group is less likely to receive the favourable prediction.

| Metric | Definition | What it asks |
|---|---|---|
| Statistical Parity Difference (SPD) | P(ŷ = retain \| unpriv) − P(ŷ = retain \| priv) | Are both groups predicted to stay at the same rate? |
| Average Odds Difference (AOD) | ½ [(FPR<sub>u</sub> − FPR<sub>p</sub>) + (TPR<sub>u</sub> − TPR<sub>p</sub>)] | Are error rates equal across groups? |
| Predictive Parity Difference (PPD) | PPV<sub>u</sub> − PPV<sub>p</sub> | Is a "will stay" prediction equally reliable for both groups? |

The three metrics encode different, partly incompatible ideas of fairness, so they are always reported together.

### Uncertainty

Every metric gets a **95% percentile bootstrap confidence interval** from 1,000 resamples of the test set. Resamples that happen to contain only one class are skipped, because AUC is undefined on them. A disparity is counted as **real only when its interval excludes zero**. This guards against reading a gap into what is only test-set composition. With 47 leavers in the test set, one employee moves recall by about 2 percentage points.

Four models × three attributes × three metrics gives **36 fairness checks**.

## 5. Bias mitigation

Both strategies come from AI Fairness 360 and use the same models and data split. Each was run separately for each protected attribute: 2 strategies × 4 models × 3 attributes = **24 mitigation runs**.

| | Reweighing | Equalised odds post-processing |
|---|---|---|
| Stage | Pre-processing | Post-processing |
| Mechanism | Assigns instance weights to training rows so that group and label become statistically independent | Adjusts group-specific decision thresholds on a trained model's outputs |
| Retraining | Yes, on reweighted data | No |
| Applied to | Training data only | Predictions |

After mitigation, every model is re-scored with the identical measurement suite. The fairness change and the performance change are then compared with that model's own baseline.

## 6. Explainability

Global feature importance is the mean absolute SHAP value per feature, computed for each model. An attribute's **SHAP rank** is the position of its best-placed feature among the 44. For example, marital status is represented by both `MaritalStatus_Married` and `MaritalStatus_Single`, and the higher-ranked of the two is used.

The ranks are then correlated with the absolute measured disparity on each fairness metric. The question is simple: if a model leans on an attribute more, is it also less fair on that attribute?

The correlation came out at −0.565 (SPD), −0.511 (AOD) and +0.268 (PPD). The metric choice flips both the sign and the strength, and gender, the attribute with no disparity, supplies most of the signal. The disparities instead travel through correlated proxies such as `JobLevel`, `MonthlyIncome`, `TotalWorkingYears` and `YearsAtCompany`.

## 7. Reproducibility

- A fixed seed (`42`) is used for the split, the models, cross-validation and the bootstrap.
- Dependencies are pinned in [`requirements.txt`](../requirements.txt) for Python 3.13. The dev container builds the same environment.
- The train/test split, fitted preprocessor and predictions are written to disk, so each stage reads the previous stage's artefacts instead of recomputing them.

## 8. Limitations

- **Fictional data.** The IBM dataset was built by IBM data scientists for analytics research. It is useful for method development, but its group patterns are not evidence about any real workforce.
- **Binarised attributes.** Age is split at 40, and marital status groups single and divorced together. Different cut-offs could change which disparities appear.
- **One attribute at a time.** Each attribute was audited and mitigated separately. Intersectional groups, such as younger single employees, were not examined.
- **Small test set.** 294 rows with 47 leavers. The confidence intervals are wide, so some real but small disparities may not reach significance.
- **Compliance is not usefulness.** A model can satisfy a fairness criterion while being operationally useless. One equalised-odds model flagged 229 of 294 employees as likely to leave. So fairness results should always be read next to performance.
