# Student Academic Performance Prediction System

A Data Mining / Machine Learning project that predicts whether a student will
**PASS** or **FAIL** based on historical academic and behavioural data, so
that educational institutions can identify at-risk students early and offer
timely interventions (remedial classes, counselling, mentorship, etc.).

## Problem

Many institutions only discover that a student is struggling after final
exam results are released, which is too late for meaningful intervention.
This project uses supervised machine learning to predict a student's
likely outcome *before* the final exam, using attendance, study habits,
family background, and other behavioural/academic indicators.

## Dataset

UCI Machine Learning Repository — **Student Performance Dataset**
(`student-por.csv`, Portuguese language course, 649 records, 33 attributes).
Full attribute description: [`data/student.txt`](data/student.txt).

## Target variable

`result` = **PASS** if final grade `G3 >= 10` (out of 20), else **FAIL**.
`G1` and `G2` (period grades) are dropped before training because they are
highly correlated with `G3` and would leak the outcome.

## Methodology

1. Load and clean the dataset (checked for missing values / duplicates).
2. Engineer the PASS/FAIL target label from `G3`; drop `G1`, `G2`, `G3`.
3. Encode categorical attributes with label encoding.
4. Split into 80% training / 20% testing (stratified on the target).
5. Train three supervised classifiers: **Logistic Regression**, **Decision
   Tree**, **Random Forest**.
6. Validate each model with **stratified 5-fold cross-validation** on the
   training set, then evaluate on the held-out test set.
7. Compare models using **accuracy, precision, recall, F1-score, and the
   confusion matrix**; select the best-performing model.

## Tools

Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn.

## Project structure

```
.
├── data/
│   ├── student-por.csv          # Portuguese course dataset (used for the model)
│   ├── student-mat.csv          # Maths course dataset (provided, not used)
│   └── student.txt              # Attribute description
├── notebook/
│   └── student_performance_model.py   # Full pipeline: load -> clean -> encode -> train -> evaluate
├── outputs/
│   ├── results_summary.json              # All numeric CV + test metrics
│   ├── confusion_matrix_*.png            # Confusion matrix per model
│   ├── feature_importance.png            # Top predictors (best model)
│   └── model_comparison.png              # Bar chart comparing all 3 models
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
cd notebook
python student_performance_model.py
```

## Results (summary)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|
| Logistic Regression | 0.769 | 0.851 | 0.882 | 0.866 |
| Decision Tree | 0.762 | 0.856 | 0.864 | 0.860 |
| **Random Forest (best)** | **0.815** | **0.852** | **0.945** | **0.897** |

See `outputs/results_summary.json` for the full breakdown (including 5-fold
cross-validation scores and confusion matrices) and the project report for
a full discussion of these results.

## Future recommendations

- Address the class imbalance (549 PASS vs 100 FAIL) with resampling
  (SMOTE) or class-weighting to improve detection of at-risk (FAIL)
  students specifically.
- Try gradient boosting models (XGBoost/LightGBM) and hyperparameter tuning
  (GridSearchCV/RandomizedSearchCV).
- Collect more recent / larger datasets, and consider a 3-class or
  regression target (predicted G3 score) for finer-grained interventions.
- Deploy the model behind a simple web dashboard for lecturers/advisers.

## Author

Data Mining course project — [add your name / student ID here]
