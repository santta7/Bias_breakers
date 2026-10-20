"""Model and data utilities for the Bias Breakers Streamlit dashboard."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATHS = {
    "Diabetes classification": PROJECT_DIR / "dataset" / "diabetes_prediction_dataset_2000.csv",
    "Insurance regression": PROJECT_DIR / "dataset" / "regression_insurance.csv",
    "Patient clustering": PROJECT_DIR / "dataset" / "clustering_patients.csv",
}
TARGETS = {"Diabetes classification": "diabetes", "Insurance regression": "charges"}

CLASSIFIERS = [
    "Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes",
    "Decision Tree", "Random Forest", "Support Vector Machine", "AdaBoost",
    "Gradient Boosting", "Bagging", "MLP Classifier",
]
REGRESSORS = [
    "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
    "Decision Tree Regressor", "Random Forest Regressor",
    "Gradient Boosting Regressor", "Support Vector Regressor", "KNN Regressor",
]


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """Load one of the fixed project datasets; no arbitrary user path is accepted."""
    if dataset_key not in DATASET_PATHS:
        raise KeyError(f"Unknown dataset: {dataset_key}")
    path = DATASET_PATHS[dataset_key]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Keep the dataset/ folder beside app.py.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"The dataset '{path.name}' contains no rows.")
    return frame


def numeric_and_categorical_columns(frame: pd.DataFrame, features: list[str] | None = None):
    data = frame[features] if features is not None else frame
    numeric = data.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in data.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """All imputers, scaling, and encoders are fit inside the training pipeline."""
    numeric, categorical = numeric_and_categorical_columns(X)
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("No usable input features remain after excluding the target.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _classifier(name: str, training_size: int, seed: int = 42):
    neighbors = max(1, min(7, training_size))
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)
    if name == "Gaussian Naive Bayes":
        return GaussianNB()
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, class_weight="balanced", min_samples_leaf=2)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=160, random_state=seed, n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=2)
    if name == "Support Vector Machine":
        return SVC(probability=True, class_weight="balanced", random_state=seed)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=seed)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=seed)
    if name == "Bagging":
        base_tree = DecisionTreeClassifier(random_state=seed, min_samples_leaf=2)
        try:
            return BaggingClassifier(estimator=base_tree, n_estimators=60, random_state=seed, n_jobs=-1)
        except TypeError:  # scikit-learn < 1.2 compatibility
            return BaggingClassifier(base_estimator=base_tree, n_estimators=60, random_state=seed, n_jobs=-1)
    if name == "MLP Classifier":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=450, early_stopping=True, random_state=seed)
    raise KeyError(f"Unsupported classifier: {name}")


def _regressor(name: str, training_size: int, seed: int = 42):
    neighbors = max(1, min(7, training_size))
    if name == "Linear Regression":
        return LinearRegression(n_jobs=-1)
    if name == "Ridge Regression":
        return Ridge(alpha=1.0)
    if name == "Lasso Regression":
        return Lasso(alpha=0.05, max_iter=5000, random_state=seed)
    if name == "Elastic Net":
        return ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed)
    if name == "Decision Tree Regressor":
        return DecisionTreeRegressor(random_state=seed, min_samples_leaf=3)
    if name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=160, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    if name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed)
    if name == "Support Vector Regressor":
        return SVR(C=10.0, epsilon=0.1, kernel="rbf")
    if name == "KNN Regressor":
        return KNeighborsRegressor(n_neighbors=neighbors, n_jobs=-1)
    raise KeyError(f"Unsupported regressor: {name}")


def _feature_quality_notes(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    missing_target = int(frame[target].isna().sum())
    duplicate_rows = int(frame.duplicated().sum())
    class_distribution = None
    if frame[target].nunique(dropna=True) < 20:
        class_distribution = frame[target].value_counts(dropna=False).to_dict()
    high_corr = []
    if pd.api.types.is_numeric_dtype(frame[target]):
        numeric = frame.select_dtypes(include=np.number)
        if target in numeric and numeric.shape[1] > 1:
            corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
            high_corr = [(str(k), float(v)) for k, v in corr.items() if pd.notna(v) and abs(v) >= 0.98]
    return {
        "missing_target_rows": missing_target,
        "duplicate_rows": duplicate_rows,
        "class_distribution": class_distribution,
        "high_numeric_target_correlations": high_corr,
    }


def evaluate_classifier(dataset_key: str, model_name: str, excluded_features: tuple[str, ...] = (), test_size: float = 0.2, seed: int = 42) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    if dataset_key not in TARGETS:
        raise ValueError("Classification is configured for the diabetes dataset in this project.")
    raw = load_dataset(dataset_key)
    target = TARGETS[dataset_key]
    quality = _feature_quality_notes(raw, target)
    # Drop exact duplicates before the split to reduce train/test overlap risk.
    frame = raw.drop_duplicates().dropna(subset=[target]).copy()
    y = frame[target]
    X = frame.drop(columns=[target] + [c for c in excluded_features if c in frame.columns], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("Select at least one feature besides the target.")
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    estimator = _classifier(model_name, len(X_train), seed)
    pipeline = Pipeline([("preprocessing", build_preprocessor(X_train)), ("model", estimator)])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    prediction_seconds = time.perf_counter() - start

    labels = list(pipeline.named_steps["model"].classes_)
    is_binary = len(labels) == 2
    weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc = np.nan
    y_score = None
    fpr = tpr = None
    try:
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)
            if is_binary:
                y_score = probabilities[:, 1]
                auc = roc_auc_score(y_test, y_score)
                fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=labels[1])
            else:
                auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted", labels=labels)
                y_score = probabilities
        elif hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_test)
            if is_binary:
                y_score = scores
                auc = roc_auc_score(y_test, scores)
                fpr, tpr, _ = roc_curve(y_test, scores, pos_label=labels[1])
    except (ValueError, TypeError):
        auc = np.nan
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    metrics = {
        "Algorithm": model_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Weighted precision": float(weighted_precision),
        "Macro precision": float(macro_precision),
        "Weighted recall": float(weighted_recall),
        "Macro recall": float(macro_recall),
        "Weighted F1-score": float(weighted_f1),
        "Macro F1-score": float(macro_f1),
        "ROC-AUC": float(auc) if pd.notna(auc) else np.nan,
        "Training time (s)": float(training_seconds),
        "Prediction time (s)": float(prediction_seconds),
        "Train accuracy": float(accuracy_score(y_train, pipeline.predict(X_train))),
        "Train rows": int(len(X_train)),
        "Test rows": int(len(X_test)),
        "Total rows used": int(len(frame)),
    }
    actual_pred = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(y_pred)})
    importance = _feature_importance(pipeline)
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "class_labels": labels,
        "actual": np.asarray(y_test),
        "predicted": np.asarray(y_pred),
        "scores": y_score,
        "fpr": fpr,
        "tpr": tpr,
        "actual_pred": actual_pred,
        "model": pipeline,
        "input_columns": list(X.columns),
        "feature_importance": importance,
        "quality": quality,
        "target_distribution": frame[target].value_counts().rename_axis("Class").reset_index(name="Count"),
    }


def evaluate_regressor(dataset_key: str, model_name: str, excluded_features: tuple[str, ...] = (), test_size: float = 0.2, seed: int = 42) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    if dataset_key != "Insurance regression":
        raise ValueError("Regression is configured for the insurance dataset in this project.")
    raw = load_dataset(dataset_key)
    target = TARGETS[dataset_key]
    quality = _feature_quality_notes(raw, target)
    # Rows with a missing target cannot be used for supervised evaluation.
    frame = raw.dropna(subset=[target]).drop_duplicates().copy()
    X = frame.drop(columns=[target] + [c for c in excluded_features if c in frame.columns], errors="ignore")
    y = frame[target]
    if X.shape[1] == 0:
        raise ValueError("Select at least one feature besides the target.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    estimator = _regressor(model_name, len(X_train), seed)
    pipeline = Pipeline([("preprocessing", build_preprocessor(X_train)), ("model", estimator)])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    prediction_seconds = time.perf_counter() - start
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "Algorithm": model_name,
        "R²": float(r2_score(y_test, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "Training time (s)": float(training_seconds),
        "Prediction time (s)": float(prediction_seconds),
        "Train R²": float(r2_score(y_train, pipeline.predict(X_train))),
        "Train rows": int(len(X_train)),
        "Test rows": int(len(X_test)),
        "Total rows used": int(len(frame)),
    }
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    actual_pred = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(y_pred), "Residual": residuals})
    return {
        "metrics": metrics,
        "actual": np.asarray(y_test),
        "predicted": np.asarray(y_pred),
        "residuals": residuals,
        "actual_pred": actual_pred,
        "model": pipeline,
        "input_columns": list(X.columns),
        "feature_importance": _feature_importance(pipeline),
        "quality": quality,
    }


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return native importance/coefficient magnitudes when available."""
    model = pipeline.named_steps.get("model")
    try:
        names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    if len(names) != len(values):
        return pd.DataFrame(columns=["Feature", "Importance"])
    return pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False).head(25)


def run_clustering(n_clusters: int = 3, seed: int = 42, include_tsne: bool = False) -> dict[str, Any]:
    from sklearn.manifold import TSNE
    frame = load_dataset("Patient clustering")
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    X_raw = frame[numeric]
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=numeric, index=frame.index)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    models = {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=seed, n_init=10),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    }
    labels_by_model = {name: model.fit_predict(X_scaled) for name, model in models.items()}
    metric_rows = []
    for name, labels in labels_by_model.items():
        metric_rows.append({
            "Algorithm": name,
            "Number of Clusters": int(len(np.unique(labels))),
            "Silhouette Score": float(silhouette_score(X_scaled, labels)),
            "Davies-Bouldin Index": float(davies_bouldin_score(X_scaled, labels)),
            "Calinski-Harabasz Score": float(calinski_harabasz_score(X_scaled, labels)),
        })
    metric_df = pd.DataFrame(metric_rows)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_scaled)
    vis = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
    for name, labels in labels_by_model.items():
        vis[name] = labels.astype(str)
    inertias = []
    k_values = list(range(2, min(9, len(frame))))
    for k in k_values:
        inertias.append(KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled).inertia_)
    tsne_coords = None
    if include_tsne:
        perplexity = min(30, max(5, (len(X_scaled) - 1) // 3))
        tsne_coords = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=seed).fit_transform(X_scaled)
    assignments = X_imputed.copy()
    for name, labels in labels_by_model.items():
        assignments[name.replace(" ", "_").replace("(", "").replace(")", "")] = labels
    return {
        "metrics": metric_df,
        "visualization": vis,
        "labels": labels_by_model,
        "scaled": X_scaled,
        "imputed": X_imputed,
        "features": numeric,
        "inertia_k": k_values,
        "inertia": inertias,
        "tsne": tsne_coords,
        "pca_variance": pca.explained_variance_ratio_,
        "assignments": assignments,
        "row_count": len(frame),
        "missing_before": int(frame.isna().sum().sum()),
    }
