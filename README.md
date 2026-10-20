# Bias Breakers — Machine Learning Capstone

An end-to-end Machine Learning project covering **Regression, Classification, and Clustering**, with exploratory data analysis, data preprocessing, model training, performance evaluation, and an interactive Streamlit dashboard.

## Project Overview

Bias Breakers explores multiple machine learning algorithms across three problem tracks. The project compares model performance using appropriate evaluation metrics and provides visualizations to support model analysis and interpretation.

### Objectives

- Perform Exploratory Data Analysis (EDA) and understand dataset characteristics.
- Handle missing values and preprocess features appropriately.
- Train and compare multiple machine learning algorithms.
- Evaluate models using relevant performance metrics.
- Visualize distributions, relationships, model performance, and clusters.
- Build an interactive web dashboard using Streamlit.

## Project Tracks

### 1. Regression

The regression track predicts a continuous target using insurance-related data.

**Algorithms implemented:**

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor

**Evaluation metrics:**
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### 2. Classification

The classification track uses diabetes-related data to predict the target class.

**Algorithms implemented:**

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine (SVM)
6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. Multilayer Perceptron (MLP) Classifier

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- Weighted F1-score
- Confusion Matrix
- ROC-AUC score, where applicable

### 3. Clustering

The clustering track explores patient data using unsupervised learning techniques.

**Algorithms implemented:**

1. K-Means Clustering
2. Agglomerative Hierarchical Clustering

**Evaluation metrics and visualizations:**
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index
- Elbow method for selecting the number of K-Means clusters
- Dendrogram for hierarchical clustering
- PCA-based two-dimensional cluster visualization
- t-SNE visualization, where implemented

## Datasets

The project includes the following datasets:

| Dataset | Purpose |
|---|---|
| `regression_insurance.csv` | Regression analysis |
| `diabetes_prediction_dataset_2000.csv` | Classification analysis |
| `clustering_patients.csv` | Patient-data clustering |

The datasets are stored in the `dataset/` directory.

## Repository Structure

```text
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
│   ├── classification5_SVM.ipynb
│   ├── classification6_RF.ipynb
│   ├── classification7_AdaBoost.ipynb
│   ├── classification8_GBC.ipynb
│   ├── classification9_Bagging.ipynb
│   ├── classification10_MLP.ipynb
│   ├── clustering1_KMeans.ipynb
│   └── clustering2_Agglomerative.ipynb
├── results/
├── models/
├── app.py
├── ml_engine.py
├── requirements.txt
└── README.md
```

The `results/`, `models/`, and Streamlit configuration directories may contain additional files depending on the current implementation.

## Technologies Used

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Scientific computing:** SciPy
- **Development environment:** Jupyter Notebook
- **Web application:** Streamlit

## Installation and Setup

### Prerequisites

- Python 3
- GitHub repository downloaded or cloned to your computer
- Internet access for installing Python dependencies

### 1. Download the repository

Download the repository as a ZIP file from GitHub and extract it, or use an existing local copy.

### 2. Open the project directory

Open the extracted `Bias_breakers` folder in Visual Studio Code.

### 3. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

### 4. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard

Ensure that `app.py`, `ml_engine.py`, the required datasets, and any other files referenced by the application are present.

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed in the terminal. It is usually:

`http://localhost:8501`

Keep the terminal running while using the dashboard.

## Interactive Dashboard

The Streamlit dashboard provides a web interface for exploring the machine learning project.

Depending on the implemented features, users can explore datasets, visualize feature distributions, train and evaluate models, compare algorithms, inspect evaluation metrics, and examine clustering results.

## Reproducibility and Evaluation

For fair comparisons, models should use consistent training and testing splits within each track. Preprocessing transformations should be fitted on the training data only to prevent data leakage.

Random states should be fixed where applicable to improve reproducibility. Model performance should be interpreted using appropriate metrics rather than relying on a single score.

## Project Team

**Project:** Bias Breakers — Machine Learning Capstone  
**Course:** 23CSE301 — Machine Learning  
**Institution:** Amrita Vishwa Vidyapeetham  
**Academic Year:** 2026–2027

Add your team members' names and any required acknowledgements here.

## Future Improvements

- Hyperparameter optimization for selected models.
- Further comparison of model performance.
- Improved interactive visualizations.
- Public deployment of the Streamlit dashboard.
- Additional model interpretability and error analysis.

## Acknowledgements

This project is developed as part of the Machine Learning Capstone coursework. External resources and AI-assisted code scaffolding should be acknowledged where required by the course guidelines.
