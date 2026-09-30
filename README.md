# Customer Churn Prediction

Predicting telecom customer churn using classical machine learning, with a focus on feature selection and rigorous model comparison rather than throwing one algorithm at the problem.

## What this does

- Cleans and preprocesses a telecom customer dataset (missing-value imputation, IQR-based outlier treatment, label/one-hot encoding)
- Runs chi-square hypothesis tests and correlation analysis to identify which features actually drive churn
- Applies **Recursive Feature Elimination (RFE)** and **Sequential Feature Selection (SFS)** to cut down the feature space before modeling
- Trains and cross-validates five classifiers: Logistic Regression, KNN, Polynomial SVC, SGD, and Random Forest
- Tunes hyperparameters with `GridSearchCV`

## Results

| Model | 5-fold CV Accuracy |
|---|---|
| Random Forest | up to 100% |
| Polynomial SVC | ~99% |
| Logistic Regression / KNN / SGD | competitive baselines |

## Tech stack

Python · pandas · scikit-learn · matplotlib/seaborn

## Notes

This was a coursework project focused on demonstrating a full ML workflow end-to-end: EDA → hypothesis testing → feature selection → model comparison → tuning. The 100% CV accuracy on Random Forest is worth treating with healthy skepticism on a small/synthetic dataset — it's documented here as-is rather than smoothed over.
