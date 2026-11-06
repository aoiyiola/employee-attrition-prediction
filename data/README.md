# Data

## Source

**IBM HR Analytics Employee Attrition & Performance**, available on [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

This is a fictional dataset created by IBM data scientists. It contains 1,470 employee records with 35 columns, and the binary target is `Attrition` (Yes/No). No real employees are represented.

## Layout

| Path | Contents | In git? |
|---|---|---|
| `raw/ibm_dataset.csv` | The original dataset, unmodified | Yes |
| `processed/cleaned_ibm_dataset.csv` | Uninformative columns removed, protected-group flags added (1,470 × 34) | No, generated |
| `processed/train_dataset.csv` | Encoded training split (1,176 × 48) | No, generated |
| `processed/test_dataset.csv` | Encoded test split (294 × 48) | No, generated |
| `processed/preprocessor.joblib` | Fitted scaler and encoder | No, generated |

Run [`notebooks/01_data_preparation.ipynb`](../notebooks/01_data_preparation.ipynb) to regenerate everything in `processed/`.

## Columns

The raw data covers four groups of information:

- **Demographics:** `Age`, `Gender`, `MaritalStatus`, `Education`, `EducationField`, `DistanceFromHome`
- **Role:** `Department`, `JobRole`, `JobLevel`, `BusinessTravel`, `OverTime`
- **Pay and tenure:** `MonthlyIncome`, `DailyRate`, `HourlyRate`, `MonthlyRate`, `PercentSalaryHike`, `StockOptionLevel`, `TotalWorkingYears`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, `NumCompaniesWorked`, `TrainingTimesLastYear`
- **Survey scores (1–4):** `EnvironmentSatisfaction`, `JobInvolvement`, `JobSatisfaction`, `PerformanceRating`, `RelationshipSatisfaction`, `WorkLifeBalance`

**Removed:** `EmployeeCount`, `Over18` and `StandardHours` are constant, and `EmployeeNumber` is an identifier.

**Added:** three binary audit columns, where 1 marks the privileged group:

| Column | 1 = | 0 = |
|---|---|---|
| `age_priv` | Age 40 and over | Under 40 |
| `gender_priv` | Male | Female |
| `marital_priv` | Married | Single or divorced |

These flags are kept out of the model inputs and used only to measure fairness. In the processed splits, `Attrition` is encoded as 1 = left and 0 = stayed.
