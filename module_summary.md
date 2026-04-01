# Module Summary Report — APA 7
## Diabetes Risk Prediction: A Machine Learning Classification Workflow

---

**Title:** Predicting Diabetes Risk Using Machine Learning: A Classification Workflow on the Pima Indians Diabetes Dataset

**Author:** Sébastien Bodrero

**Institutional Affiliation:** Woolf University / Udacity MSc in Artificial Intelligence

**Course:** AI Mastery — Module 3: Machine Learning Foundations

**Date:** April 2026

---

## Overview

This report documents the design, implementation, and findings of a supervised machine learning workflow applied to the Pima Indians Diabetes Dataset (768 rows × 9 columns). The task is binary classification: predict whether a female patient of Pima Indian heritage has diabetes based on eight diagnostic measurements. Two models were built and compared: a Logistic Regression baseline and a Random Forest primary classifier. As Adhikari (2022) notes, a reproducible workflow "ensures that others can verify, build on, and extend Sebastien Bodrero analysis" — both automated download of the dataset and pinned dependencies via `requirements.txt` are implemented to this end. The dataset was sourced from the Jason Brownlee Datasets GitHub mirror and is originally from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) via the UCI Machine Learning Repository.

---

## Dataset Description

The **Pima Indians Diabetes Dataset** contains 768 records of female patients aged 21 and above of Pima Indian heritage — a population with one of the highest documented rates of Type 2 diabetes. The dataset contains **8 numeric feature columns** and **1 binary target column** (`Outcome`), with no explicit missing value markers. However, five features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) contain physiologically impossible zero values that encode missing data — a widely noted data quality characteristic of this dataset. The target class is moderately imbalanced: 500 patients (65.1%) are non-diabetic and 268 (34.9%) are diabetic. The dataset is not the same as those used in Module 1 (NYC Airbnb 2019) or Module 2 (Medical Insurance Costs) and is publicly available for academic use.

---

## Workflow Description

The analysis was implemented in six sequential sections in `modeling.ipynb`:

### 1. Setup

Standard scientific Python libraries were imported: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`. A consistent plot theme was applied via `sns.set_theme()`. Library versions were printed for reproducibility.

### 2. Data Ingestion

The dataset was auto-downloaded using `urllib.request` if not already present locally. Column names were assigned manually (the raw file has no header row). The loaded DataFrame was inspected with `.head()`, `.info()`, and `.describe()`. The key observation was that minimum values of 0 in five columns indicate missing data.

### 3. Data Preparation & Preprocessing

Three steps were applied:

- **Zero-to-NaN replacement:** Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were replaced with `NaN` (652 cells affected, 17.0% of those columns). `Pregnancies` and `Age` were left unchanged as their zero/minimum values are physiologically valid.
- **Median imputation:** Each affected column was imputed with its own median. Median was chosen over mean because the distributions are right-skewed (particularly `Insulin` with 48.7% missing and `SkinThickness` with 29.6% missing).
- **Train-test split + scaling:** An 80/20 stratified split produced 614 training and 154 test rows with preserved class ratios. `StandardScaler` was fit on training data only and applied to both sets to prevent data leakage (Pedregosa et al., 2011).

### 4. Model Selection & Training

Two models were trained on the same preprocessed data:

- **Logistic Regression** (`solver='lbfgs'`, `max_iter=1000`): Linear baseline classifier, computationally efficient and interpretable.
- **Random Forest** (`n_estimators=200`, `random_state=42`): Ensemble of 200 decision trees with bootstrap sampling and random feature subsets at each split. Random forests consistently outperform single linear classifiers on tabular data by capturing non-linear feature interactions (Breiman, 2001).

### 5. Evaluation

Both models were evaluated on the held-out test set using five metrics: accuracy, precision, recall, F1-score, and ROC-AUC. Three visualisations were produced:

1. **Confusion matrices** (side-by-side heatmaps) — showing the distribution of true positives, true negatives, false positives, and false negatives for each model.
2. **ROC curves** — plotting the true positive rate vs. false positive rate at all classification thresholds for both models simultaneously.
3. **Feature importance bar chart** (Random Forest) — ranking all eight features by their mean decrease in impurity across the 200 trees.

### 6. Notebook Summary

A 4–6 sentence synthesis in the final notebook cell covering the problem, model choice, key metrics, and the primary challenge (class imbalance driving metric selection).

---

## Key Decisions and Assumptions

| Decision | Justification |
|---|---|
| Median imputation for zero-encoded missing values | Right-skewed distributions (Insulin, SkinThickness) make median more robust than mean |
| Stratified 80/20 train-test split | Preserves 34.9% positive class ratio in both subsets |
| Scaler fit on training data only | Prevents data leakage — test set treated as genuinely unseen |
| F1-score and ROC-AUC as primary metrics | Class imbalance (65/35) makes accuracy misleading; recall prioritised for clinical context |
| Random Forest as primary model | Captures non-linear interactions; provides feature importance; robust to outliers (Breiman, 2001) |
| n_estimators=200 | Sufficient trees for stable importance estimates without excessive compute |
| Default threshold (0.5) | Used for standard comparison; lowering to ~0.35 would increase recall at cost of precision |

---

## Results and Interpretation

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.7078 | **0.7403** |
| Precision | 0.6000 | **0.6522** |
| Recall | 0.5000 | **0.5556** |
| F1-score | 0.5455 | **0.6000** |
| ROC-AUC | 0.8130 | **0.8173** |

The Random Forest outperforms Logistic Regression across all five metrics. The F1-score improvement (+0.0545, ~10% relative) confirms that non-linear feature interactions — particularly between Glucose, BMI, Age, and DiabetesPedigreeFunction — provide genuine discriminative signal beyond what a linear model can capture.

The ROC-AUC values (0.813 / 0.817) are notably higher than accuracy, reflecting the class imbalance. Both models can distinguish diabetic from non-diabetic patients far better than random chance, but at the default threshold of 0.5 they prioritise precision over recall, missing approximately 44–50% of diabetic patients. In a clinical screening deployment, lowering the threshold to ~0.35 would substantially improve recall.

**Feature importance (Random Forest):** Glucose dominates (26.9%), followed by BMI (15.8%), DiabetesPedigreeFunction (12.5%), and Age (12.0%). BloodPressure (8.4%) and SkinThickness (7.3%) contribute least — the latter likely depressed by its high rate of imputed values.

---

## Responsible Practice: Bias and Ethical Considerations

1. **Population-specific model:** The dataset covers exclusively Pima Indian women aged 21+, a population with unusually high genetic susceptibility to Type 2 diabetes. Deploying this model in a different demographic without retraining would risk systematic misclassification — a form of algorithmic unfairness.

2. **Class imbalance under-referral risk:** The low recall at the default threshold (~56% for Random Forest) means the model fails to flag approximately 44% of diabetic patients. Deploying with the default threshold in a clinical screening context could cause systematic under-referral. Threshold calibration to the acceptable false-negative rate is required before any real-world use.

3. **Imputation masking true signal:** With 48.7% of Insulin values imputed to the median, the true predictive signal of insulin is partially obscured. This could lead to underestimating the importance of insulin in more complete datasets.

4. **Transparency and accountability:** Any clinical use of a black-box model like Random Forest must be accompanied by explainability tools (e.g., SHAP values) and human-in-the-loop review. The feature importance reported here is a starting point, not a full explanation.

---

## Reproducibility

The notebook is designed to be fully reproducible by any user with Python 3.10+ and internet access:

- **Automated download:** The ingestion cell checks for `diabetes.csv` locally and downloads it if absent.
- **Pinned dependencies:** All package versions are recorded in `requirements.txt` (generated via `pip freeze`).
- **Fixed random states:** `random_state=42` is set in `train_test_split`, `LogisticRegression`, and `RandomForestClassifier` to ensure identical results across runs.
- **Top-to-bottom execution:** The notebook executes without errors from a clean kernel, validated using `jupyter nbconvert --to notebook --execute`.

Adhikari (2022) identifies dependency pinning and automated execution checks as foundational requirements for reproducibility in data science projects — both are implemented here.

---

## References

Adhikari, N. K. J. (2022). *Reproducible data science with Python: An open learning resource*. ResearchGate. https://doi.org/10.13140/RG.2.2.22099.04641

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
