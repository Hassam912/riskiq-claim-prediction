# RiskIQ — Auto Insurance Claim Prediction

> A claim-likelihood model built around the *pricing decision* it exists to serve — plus a
> deployable scoring function that turns a raw intake form into a risk tier.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4c72b0?style=flat)

📄 **[Full write-up — the business case, model selection reasoning, and deployment](https://hassamasghar.com/projects/car-insurance-claim-predictor)**

---

## At a glance

| | |
|---|---|
| **Context** | MMA 867 — Predictive Modelling, Queen's Smith School of Business |
| **My role** | Solution architect — EDA, model interpretation, and the deployable underwriting tool |
| **Problem type** | Binary classification, moderately imbalanced (31/69) |
| **Scale** | 10,000 policyholder records × 18 variables, 4 models benchmarked |
| **Headline result** | **0.887 ROC-AUC** — and the *most interpretable* model won |

---

## The business problem

Blanket, demographic-class premiums assume risk shifts slowly by group — age band, postal
code, vehicle class — and charge everyone in a class the same. That assumption broke.
Canadian insurers on blanket pricing were absorbing losses **averaging 18% more in claims
and legal overhead than they collected in premiums**, a gap running to roughly **$1.2B in
Alberta alone** in a recent fiscal year.

Blanket pricing also self-selects against the insurer:

- Overcharge a safe driver relative to their real risk → they leave for a competitor pricing them correctly.
- Undercharge a risky one → they stay.

The book quietly gets worse every renewal cycle. A **per-policy risk score at quote time**
is the direct countermeasure.

## Approach

**Missingness treated as signal, not noise.** ~10% of records were missing `credit_score`
and `annual_mileage`. Rather than drop rows or silently impute, a **`credit_score_missing`
flag** was added before median-imputing — thin or absent credit history is itself a risk
signal that a silently-imputed median would erase.

**Four models, one honest comparison** on an identical split, scored on precision, recall,
F1 and ROC-AUC — never accuracy alone (a model predicting "no claim" for everyone still
scores ~69%).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.831** | **0.732** | **0.726** | **0.729** | **0.8865** |
| Gradient Boosting | 0.830 | 0.731 | 0.723 | 0.727 | 0.8814 |
| Random Forest | 0.825 | 0.726 | 0.707 | 0.716 | 0.8762 |
| Decision Tree | 0.825 | 0.726 | 0.710 | 0.718 | 0.8729 |

**Logistic regression won outright.** That matters well beyond the leaderboard:
interpretability isn't a nice-to-have in insurance pricing, it's close to a legal
requirement. A regulator asking why a specific customer was priced a certain way needs a
better answer than "the gradient boosting model said so." Here, no accuracy was traded
away to get explainability — the same model delivered both.

## What actually drives claim risk

Standardised logistic coefficients double as the interpretability layer. Negative =
*lowers* claim odds:

| Driver | Coefficient | Odds ratio |
|---|---|---|
| Driving experience | **−1.687** | 0.19 |
| Vehicle year < 2015 | +0.776 | 2.17 |
| Vehicle ownership (owned) | −0.770 | 0.46 |
| Gender | +0.455 | 1.58 |
| Past accidents | −0.370 | 0.69 |
| Speeding violations | +0.188 | 1.21 |

Driving experience dominates by a wide margin — ahead of anything about the vehicle
itself.

## The deployable piece

A notebook that produces a good ROC-AUC and stops is a homework assignment. What makes
this a *solution* is `score_customer()` — it takes a record in the shape an intake form
actually produces (human-readable strings, possibly-missing fields) and returns a
probability and a risk tier:

```python
# Young, inexperienced, no credit history on file, sports car, prior incidents
score_customer({
    'age': 0, 'gender': 1, 'driving_experience': '0-9y',
    'education': 'high school', 'income': 'working class',
    'credit_score': None,              # handled: flag + training-median impute
    'vehicle_ownership': 0.0, 'vehicle_year': 'before 2015',
    'married': 0.0, 'children': 0.0, 'annual_mileage': 18000,
    'vehicle_type': 'sports car',
    'speeding_violations': 3, 'duis': 1, 'past_accidents': 2,
})
# → {'claim_probability': 0.9391, 'risk_tier': 'Very High', 'recommend_claim_label': 1}

# Experienced, owns the car, good credit, clean record
# → {'claim_probability': 0.0085, 'risk_tier': 'Low', 'recommend_claim_label': 0}
```

Risk tiers are equal-width bands (`Low` < 0.25, `Medium` < 0.50, `High` < 0.75, else
`Very High`). Those cut points are placeholders — **in production they should be set from
the relative cost of the two error types**, not split evenly. Underpricing a genuinely
risky policy costs a claim payout; overpricing a safe one costs a customer to a competitor.

It applies **the same** ordinal/one-hot encoding and imputation logic the training
pipeline used, so a production score can never silently drift from how the model was fit.
Wrapped behind a form intake, it's deployable as-is.

## Repo contents

| File | What's in it |
|---|---|
| `riskiq_claim_prediction.ipynb` | Full pipeline — EDA, cleaning, feature engineering, hypothesis testing, 4-model benchmark, feature importance, `score_customer()` |
| `car_insurance.csv` | The underlying 10,000-record policyholder dataset |

## Running it

```bash
pip install pandas numpy scikit-learn scipy seaborn matplotlib
jupyter notebook riskiq_claim_prediction.ipynb
```

Data loads from the repo directory — no path edits needed. Runs top-to-bottom in under a
minute.

## Limitations

- **Not a live book of business.** The dataset is realistic but needs validation against
  real claims experience before any premium decision leans on it.
- **Static snapshot, not a monitored model.** Risk profiles drift; production needs
  scheduled retraining and drift monitoring.
- **Fairness review unfinished.** `gender` carries a meaningful coefficient. Any model
  pricing on demographic-adjacent features needs an explicit fairness and
  regulatory-compliance pass before it touches a real quote.

---

*Course project for MMA 867, Queen's Smith School of Business. I served as solution
architect — this repository is the analysis and tooling I owned.*
