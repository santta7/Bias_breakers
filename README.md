# Bias Breakers – Machine Learning Algorithms

## Overview

This repository contains hands-on implementations and experimentation with supervised machine learning algorithms, covering both **regression** and **classification**. Each algorithm is implemented in its own Jupyter notebook, following a consistent workflow of exploratory data analysis, preprocessing, model training, hyperparameter tuning, and evaluation.

## Table of Contents

- [Project Structure](#project-structure)
- [Regression](#regression)
- [Classification](#classification)
- [Dataset](#dataset)
- [Technologies / Libraries](#technologies--libraries)
- [Installation](#installation)
- [Usage](#usage)
- [Learning Objectives](#learning-objectives)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Structure

```
Bias_breakers/
├── dataset/
│   ├── clustering_patients.csv
│   ├── diabetes_prediction_dataset_2000.csv
│   └── regression_insurance.csv
├── notebooks/
│   ├── Regression01_LinearRegression.ipynb
│   ├── Regression02_ridgeRegression.ipynb
│   ├── Regression03_lassoRegression.ipynb
│   ├── Regression04_elasticNetRegression.ipynb
│   ├── Regression05_polynomialRegression.ipynb
│   ├── Regression06_DecisionTree.ipynb
│   ├── Regression07_RandomForest.ipynb
│   ├── Regression08_GradientBoosting.ipynb
│   ├── Regression09_SupportVector.ipynb
│   ├── Regression10_KNN.ipynb
│   ├── classification1_LR.ipynb
│   ├── classification2_KNN.ipynb
│   ├── classification3_GNB.ipynb
│   ├── classification4_DT.ipynb
│   └── classification5_SVM.ipynb
├── requirements.txt
└── README.md
```

## Regression

All ten regression notebooks predict `charges` (medical insurance cost) from the `regression_insurance.csv` dataset. Each notebook follows the same pipeline: exploratory data analysis, IQR-based outlier capping on the input features, a `bmi_age` interaction feature, one-hot encoding of categorical columns (`sex`, `smoker`, `region`) with `pandas.get_dummies`, feature scaling with `StandardScaler`, hyperparameter search with `GridSearchCV`/`cross_val_score`, and evaluation with R², MSE, RMSE, and MAE.

| # | Notebook | Algorithm | Problem Type | Key Concepts Demonstrated |
|---|----------|-----------|---------------|----------------------------|
| 1 | `Regression01_LinearRegression.ipynb` | Linear Regression | Regression | Ordinary least squares fitting; establishes the baseline model with no hyperparameters, used as the reference point for the regularized variants |
| 2 | `Regression02_ridgeRegression.ipynb` | Ridge Regression | Regression | L2 regularization; shrinking coefficients to reduce overfitting/multicollinearity |
| 3 | `Regression03_lassoRegression.ipynb` | Lasso Regression | Regression | L1 regularization; automatic feature selection by shrinking weak coefficients to exactly zero |
| 4 | `Regression04_elasticNetRegression.ipynb` | Elastic Net Regression | Regression | Combined L1/L2 penalty; tuning the `l1_ratio` to balance the two penalties |
| 5 | `Regression05_polynomialRegression.ipynb` | Polynomial Regression | Regression | `PolynomialFeatures` to capture non-linear relationships and feature interactions; selecting the polynomial degree via cross-validation |
| 6 | `Regression06_DecisionTree.ipynb` | Decision Tree Regression | Regression | Non-linear, rule-based splitting; capturing feature interactions without manual feature engineering |
| 7 | `Regression07_RandomForest.ipynb` | Random Forest Regression | Regression | Ensemble of decision trees (bagging) to reduce variance and improve generalization |
| 8 | `Regression08_GradientBoosting.ipynb` | Gradient Boosting Regression | Regression | Sequential ensemble (boosting) that fits new trees to the residual errors of prior trees |
| 9 | `Regression09_SupportVector.ipynb` | Support Vector Regression | Regression | Kernel-based regression with an epsilon-insensitive margin; permutation importance for feature analysis |
| 10 | `Regression10_KNN.ipynb` | KNN Regression | Regression | Distance-based, instance-based learning; predicting from the average of nearest neighbors; permutation importance for feature analysis |

## Classification

All five classification notebooks predict `diabetes` (binary target) from the `diabetes_prediction_dataset_2000.csv` dataset. Each notebook follows the same pipeline: exploratory data analysis, IQR-based outlier checks on `bmi` and `blood_glucose_level`, a `glucose_hba1c_interaction` engineered feature, a `ColumnTransformer`/`Pipeline` with `OneHotEncoder` (categorical columns) and `StandardScaler` (numeric columns), a stratified train/test split, and evaluation with accuracy, precision, recall, weighted F1, ROC-AUC, a confusion matrix, and an ROC curve. Each notebook saves its metrics into a shared results file to build the final comparison table.

| # | Notebook | Algorithm | Problem Type | Key Concepts Demonstrated |
|---|----------|-----------|---------------|----------------------------|
| 1 | `classification1_LR.ipynb` | Logistic Regression | Binary Classification | Linear decision boundary via the logistic (sigmoid) function; probability-based predictions |
| 2 | `classification2_KNN.ipynb` | KNN Classification | Binary Classification | Distance-based classification by majority vote of nearest neighbors |
| 3 | `classification3_GNB.ipynb` | Gaussian Naive Bayes | Binary Classification | Probabilistic classification assuming feature independence and Gaussian-distributed features |
| 4 | `classification4_DT.ipynb` | Decision Tree Classification | Binary Classification | Rule-based, non-linear decision boundaries; feature importance from tree splits |
| 5 | `classification5_SVM.ipynb` | Support Vector Machine Classification | Binary Classification | Margin-maximizing, kernel-based classification; combines the five models' saved results into a final comparison table |

## Dataset

| File | Description | Used In |
|------|-------------|---------|
| `regression_insurance.csv` | Medical insurance records with `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the target `charges` | All 10 regression notebooks |
| `diabetes_prediction_dataset_2000.csv` | Patient records with `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`, and the binary target `diabetes` | All 5 classification notebooks |
| `clustering_patients.csv` | Patient records with `age`, `blood_pressure_sys`, `cholesterol`, `sodium_potassium_ratio`, and `heart_rate` | Present in the dataset folder but not used by any notebook in this repository yet |

## Technologies / Libraries

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Installation

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
```

Windows activation:

```bash
.venv\Scripts\activate
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment and install the dependencies as shown above.
2. Launch Jupyter from the project root:

   ```bash
   jupyter notebook
   ```

3. Open any notebook in the `notebooks/` folder and run the cells in order (top to bottom). Each notebook reads its dataset directly from the `dataset/` folder using a relative path, so no extra configuration is needed.

## Learning Objectives

This project is designed to build a practical understanding of:

- Supervised learning workflows (regression and classification)
- Train/test splitting, including stratified splitting for imbalanced classification targets
- Feature preprocessing: outlier handling, feature engineering (interaction features), one-hot encoding, and feature scaling
- Model training and hyperparameter tuning with `GridSearchCV` and cross-validation
- Regularization (L1, L2, and combined Elastic Net penalties)
- Ensemble methods (Random Forest, Gradient Boosting)
- Distance-based learning (KNN)
- Kernel-based learning (SVM/SVR)
- Model evaluation using appropriate metrics for regression (R², MSE, RMSE, MAE) and classification (accuracy, precision, recall, F1, ROC-AUC)

## Future Improvements

The following are proposed ideas and are **not currently implemented** in this repository:

- Extend the project to use `clustering_patients.csv` for an unsupervised learning (clustering) module
- Add a unified script or notebook that automatically aggregates and visualizes results across all regression models, similar to the existing classification comparison table
- Add automated tests and CI to validate notebook execution
- Experiment with additional feature engineering and advanced hyperparameter search strategies (e.g., randomized or Bayesian search)

## License

This project does not currently include a license file.
