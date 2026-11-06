"""
Student Academic Performance Prediction System
================================================
Data Mining / Machine Learning course project

Goal: Predict whether a student will PASS or FAIL (based on final grade G3)
using historical academic and behavioural data from the UCI "Student
Performance" dataset (Portuguese language course, student-por.csv, 649
records).

Tools used:
- Python 3
- pandas, numpy          -> data loading & preprocessing
- scikit-learn            -> encoding, train/test split, models, cross-validation, metrics
- matplotlib, seaborn     -> confusion matrix / comparison plots

Pipeline (matches the methodology defined in BACKGROUND_OF_THE_PROBLEM.docx):
1. Load dataset
2. Clean data (check missing values / duplicates)
3. Create target label: PASS (G3 >= 10) / FAIL (G3 < 10)
4. Drop G1 and G2 (they are strong leakage predictors of G3, so excluding
   them keeps the model genuinely predictive rather than trivially correct)
5. Encode categorical variables (Label/One-Hot Encoding)
6. Split data: 80% train / 20% test (stratified on the target)
7. Train 3 supervised classifiers: Logistic Regression, Decision Tree, Random Forest
8. Evaluate each with stratified 5-fold cross-validation + hold-out test metrics
   (accuracy, precision, recall, F1-score, confusion matrix)
9. Select the best-performing model
10. Save results/plots to the outputs/ folder
"""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
df = pd.read_csv("../data/student-por.csv", sep=";")
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

# ---------------------------------------------------------------
# 2. Data cleaning checks
# ---------------------------------------------------------------
print("\nMissing values per column:\n", df.isnull().sum().sum(), "total missing cells")
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------------
# 3. Create target label: PASS / FAIL based on final grade G3
#    (UCI convention: G3 >= 10 out of 20 = PASS)
# ---------------------------------------------------------------
df["result"] = df["G3"].apply(lambda g: "PASS" if g >= 10 else "FAIL")
print("\nClass balance:\n", df["result"].value_counts())

# ---------------------------------------------------------------
# 4. Drop G1, G2, G3 (G1/G2 leak the final outcome; G3 was used only to build the label)
# ---------------------------------------------------------------
features_df = df.drop(columns=["G1", "G2", "G3", "result"])
target = df["result"]

# ---------------------------------------------------------------
# 5. Encode categorical variables
# ---------------------------------------------------------------
categorical_cols = features_df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    features_df[col] = le.fit_transform(features_df[col])
    encoders[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))

le_target = LabelEncoder()
y = le_target.fit_transform(target)  # FAIL=0, PASS=1
X = features_df

print("\nEncoded feature matrix shape:", X.shape)

# ---------------------------------------------------------------
# 6. Train / test split (80 / 20, stratified)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 7 & 8. Train & evaluate 3 models with 5-fold stratified CV + hold-out test
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1"]

results_summary = {}

for name, model in models.items():
    print(f"\n================  {name}  ================")

    # 5-fold stratified cross-validation on the training set
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_means = {m: float(np.mean(cv_results[f"test_{m}"])) for m in scoring}
    cv_stds = {m: float(np.std(cv_results[f"test_{m}"])) for m in scoring}
    print("5-fold CV (train set) means:", {k: round(v, 3) for k, v in cv_means.items()})

    # Fit on full training set, evaluate on held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print("Hold-out test metrics:", {k: round(v, 3) for k, v in test_metrics.items()})

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted) [FAIL, PASS]:\n", cm)
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    results_summary[name] = {
        "cv_means": cv_means,
        "cv_stds": cv_stds,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
    }

    # Save confusion matrix plot
    plt.figure(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"../outputs/confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

# ---------------------------------------------------------------
# 9. Select best-performing model (by hold-out F1-score)
# ---------------------------------------------------------------
best_model_name = max(results_summary, key=lambda m: results_summary[m]["test_metrics"]["f1"])
print(f"\nBest performing model (by test F1-score): {best_model_name}")

# ---------------------------------------------------------------
# Feature importance for the best tree-based model (if applicable)
# ---------------------------------------------------------------
if best_model_name in ("Random Forest", "Decision Tree"):
    best_model = models[best_model_name]
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 feature importances:\n", importances.head(10))

    plt.figure(figsize=(6, 4))
    importances.head(10).sort_values().plot(kind="barh", color="#4C72B0")
    plt.title(f"Top 10 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig("../outputs/feature_importance.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------
comp_df = pd.DataFrame({name: res["test_metrics"] for name, res in results_summary.items()}).T
comp_df.plot(kind="bar", figsize=(7, 4.5))
plt.title("Model Comparison - Hold-out Test Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../outputs/model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Save full numeric results for reporting
# ---------------------------------------------------------------
with open("../outputs/results_summary.json", "w") as f:
    json.dump({"best_model": best_model_name, "results": results_summary,
               "class_balance": target.value_counts().to_dict(),
               "train_size": int(X_train.shape[0]), "test_size": int(X_test.shape[0])}, f, indent=2)

print("\nAll outputs saved to ../outputs/")
