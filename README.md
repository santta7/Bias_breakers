# Bias Breakers - Machine Learning Project

## About the Project

This repository contains our Machine Learning capstone project for Review 1.

The main aim of the project is to understand and implement different machine learning algorithms and compare their performance on suitable datasets. We are working on both classification and regression problems and implementing multiple algorithms for each.

The project is being developed step by step, with separate notebooks for the different algorithms.

---

## Project Work

### 1. Classification

For the classification part, we are using a diabetes prediction dataset.

The following five classification algorithms are implemented:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

The models are evaluated using different classification metrics, including:

- Accuracy
- Precision
- Recall
- Weighted F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

The purpose is to compare the performance of the five algorithms and understand how each model behaves on the diabetes prediction problem.

---

### 2. Regression

For the regression part, we are using an insurance dataset to predict insurance costs.

The following ten regression algorithms are included:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor (KNN Regressor)

The models are evaluated using appropriate regression metrics and their results will be compared to identify the better-performing models.

---

## Datasets

The project uses the following datasets for Review 1:

| Dataset | Purpose |
|---|---|
| `diabetes_prediction_dataset_2000.csv` | Diabetes classification |
| `regression_insurance.csv` | Insurance cost regression |

The datasets are stored inside the `dataset` folder.

---

## Project Structure

```text
Bias_breakers/
│
├── dataset/
│   ├── clustering_patients.csv
│   ├── diabetes_prediction_dataset_2000.csv
│   └── regression_insurance.csv
│
├── notebooks/
│   ├── classification1_LR.ipynb
│   ├── classification2_KNN.ipynb
│   ├── classification3_GNB.ipynb
│   ├── classification4_DT.ipynb
│   ├── classification5_SVM.ipynb
│   │
│   ├── linearRegression.ipynb
│   ├── ridgeRegression.ipynb
│   ├── lassoRegression.ipynb
│   ├── elasticNetRegression.ipynb
│   ├── polynomialRegression.ipynb
│   ├── 06_decision_tree_regressor.ipynb
│   ├── 07_random_forest_regressor.ipynb
│   ├── 08_gradient_boosting_regressor.ipynb
│   ├── 09_svr.ipynb
│   └── 10_knn_regressor.ipynb
│
├── requirements.txt
└── .gitignore
