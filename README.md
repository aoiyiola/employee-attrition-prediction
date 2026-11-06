# Fairness Audit of Employee Attrition Models

**Companies use machine learning to predict which employees are likely to leave. But do those predictions treat everyone fairly?**

This project builds four models that predict employee attrition, meaning which employees are likely to leave. It then tests whether the models treat people differently because of their **age, gender or marital status**. It also tries two ways of correcting unfair treatment and measures what each one costs in accuracy.

**Tools:** Python · scikit-learn · XGBoost · AI Fairness 360 · SHAP · pandas · Matplotlib · Jupyter

---

## Key findings

- **The models were unfair, mostly to younger employees.** I ran 36 fairness checks in total. 11 found a real difference in treatment, and every one of those disadvantaged the less advantaged group, mainly employees under 40 and unmarried employees. All four models showed this to some degree.
- **The easiest model to explain was no fairer.** Logistic Regression, one of the simplest models, had the largest gap. Employees under 40 were about **21 percentage points** less likely than older colleagues to be predicted to stay.
- **Fixing unfairness costs accuracy.** One fix removed 6 of the 11 problems and cost almost nothing in accuracy. The other removed all 11 but cost about **8 times as much** accuracy.
- **Explanation tools didn't reveal the unfairness.** Tools that show which factors a model relies on most did not point to where it was being unfair. Much of the unfairness came through related factors such as job level, income and years of experience, which go hand in hand with age.
- **Passing a fairness test is not the same as being useful.** One "corrected" model met the fairness standard only by predicting that 229 of 294 employees would leave.

![How fairly each model treated each group](results/figures/01_baseline_fairness.png)

<sub>**How to read this chart:** each row is one model, grouped by age, marital status and gender. A dot left of the centre line means the younger, unmarried or female group was treated worse. Solid dots are differences too large to be down to chance. Faded dots could be chance.</sub>

---

## Why it matters

Prediction models learn from past records. If past HR decisions were uneven, a model can repeat that pattern at scale without anyone noticing. In the UK, treating staff differently because of age, sex or marital status can breach the **Equality Act 2010**. Unfair models are therefore a legal risk as well as an ethical one.

## What I did

**1. Prepared the data.** I used 1,470 employee records from IBM's HR Analytics dataset. It is synthetic data built to resemble a real workforce, and 16% of the employees left. I cleaned the data, set 20% aside for testing, and trained the models only on the rest.

**2. Built four prediction models.** Two are easy to explain and two are complex "black box" models. That mix lets me test whether easier-to-explain models are also fairer.

| Model | How it works, in plain terms | Easy to explain? |
|---|---|---|
| Logistic Regression | Gives each factor a weight and adds them up | Yes |
| Decision Tree | Asks a series of yes/no questions | Yes |
| Random Forest | Hundreds of decision trees vote on the answer | No |
| XGBoost | Builds trees one after another, each correcting the previous one's mistakes | No |

Because only 16% of employees left, the models were set to give extra weight to the leavers so they didn't simply ignore them.

**3. Checked fairness.** I compared under-40s with over-40s, women with men, and unmarried with married employees. Each comparison used three common definitions of fairness:
- Is each group equally likely to be predicted to stay?
- Does the model make mistakes at the same rate for each group?
- When the model says someone will stay, is it equally reliable for each group?

Four models, three groups and three definitions make 36 checks. A difference only counted if it was too large to be down to chance.

**4. Tried two fixes.**
- **Reweighing** adjusts the data before the model learns, so that each group carries fair weight.
- **Equalised odds** adjusts the model's decisions afterwards, using a different decision cut-off for each group.

**5. Tested an explanation tool.** SHAP shows which factors a model relies on most. I checked whether it points to the unfairness that the fairness checks found.

For technical readers, the full method, metric definitions and limitations are in **[docs/methodology.md](docs/methodology.md)**.

---

## Results

### How well the models spotted leavers

| Model | Share of leavers correctly identified |
|---|---|
| Logistic Regression | **74%** |
| XGBoost | 44% |
| Random Forest | 35% |
| Decision Tree | 28% |

Plain accuracy is misleading here. A model that predicts "nobody leaves" would be 84% accurate and completely useless. So the models were judged mainly on how many actual leavers they caught.

### What fixing unfairness costs

| Fix | Unfair results removed | Accuracy lost |
|---|---|---|
| Reweighing (adjust the data before learning) | 6 of 11 | About half a percentage point |
| Equalised odds (adjust decisions afterwards) | 11 of 11 | About 4 percentage points |

![Fairness gained against accuracy lost, for each fix](results/figures/04_fairness_performance_tradeoff.png)

<sub>**How to read this chart:** each point is one fix applied to one group. Points further right are fairer. Points lower down lost more accuracy. The shaded corner, top right, means fairer at no cost.</sub>

### Can explanation tools spot unfairness?

![Do the factors a model relies on most show where it is unfair?](results/figures/09_shap_vs_fairness_all_metrics.png)

<sub>**How to read this chart:** each point is one model and one group. Further left means the model relied on that characteristic more. Higher up means it treated that group more unfairly. If explanation tools revealed unfairness, the points would line up. They don't, and the pattern changes depending on which fairness definition you use.</sub>

<details>
<summary><b>More figures</b></summary>

| Figure | File |
|---|---|
| Fairness after reweighing | [02_reweighing_fairness.png](results/figures/02_reweighing_fairness.png) |
| Fairness after equalised odds | [03_equalised_odds_fairness.png](results/figures/03_equalised_odds_fairness.png) |
| Factors each model relies on most | [05_shap_importance_all_models.png](results/figures/05_shap_importance_all_models.png) |
| Same, easy-to-explain models only | [06_shap_importance_transparent.png](results/figures/06_shap_importance_transparent.png) |
| Same, black-box models only | [07_shap_importance_ensemble.png](results/figures/07_shap_importance_ensemble.png) |
| Explanation tool vs one fairness definition | [08_shap_vs_fairness_spd.png](results/figures/08_shap_vs_fairness_spd.png) |

</details>

---

## What this means for organisations

1. **Check models for fairness before using them.** These problems did not show up in normal accuracy reports.
2. **Use more than one definition of fairness.** The definitions disagree, and relying on one can hide a problem.
3. **Report what each fairness fix costs.** "Fairer" means little without knowing how much accuracy was given up.
4. **Don't rely on explanation tools to find unfairness.** Bias can work through related factors that those tools don't flag.

## Skills demonstrated

- **Data preparation:** checking data quality, cleaning, and keeping test data completely separate from training.
- **Predictive modelling:** building and comparing four machine learning models on an uneven dataset where only 16% of employees left.
- **Statistical thinking:** only counting a difference as real when it is too large to be down to chance.
- **Responsible AI:** measuring and correcting bias with IBM's AI Fairness 360, and testing explanation tools with SHAP.
- **Communication:** turning technical results into clear charts and practical recommendations.
- **Reproducible work:** shared, reusable measurement code, fixed settings and a ready-to-run environment.

---

## Repository structure

```
employee-attrition-prediction/
├── data/                    # the dataset and a description of it
├── notebooks/
│   ├── 01_data_preparation.ipynb   # cleaning the data and splitting it for testing
│   └── 02_model_building.ipynb     # building the four models and measuring them
├── src/
│   └── metrics.py           # the fairness and accuracy measurements, shared by every stage
├── results/
│   └── figures/             # all charts in this README
├── docs/
│   └── methodology.md       # full technical write-up
└── requirements.txt         # software needed to run the project
```

## Running it

You need Python 3.13.

```bash
git clone https://github.com/aoiyiola/employee-attrition-prediction.git
cd employee-attrition-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then open the notebooks in order, starting with `01_data_preparation`. Alternatively, open the project in GitHub Codespaces or VS Code and choose **Reopen in Container**, which sets everything up automatically.

## Project status

The data preparation and model building notebooks are included. The notebooks for the two fixes and the explanation analysis, which produced figures 02–09, are being added.

---

## About

By **Afees Iyiola** as the research project for the MSc in Applied Artificial Intelligence and Data Analytics at the University of Bradford (2025/26).

**Data:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), a synthetic dataset created by IBM data scientists for analytics research.
