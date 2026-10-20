"""Small smoke tests for the actual bundled datasets and model workflows."""
import unittest
import numpy as np

from ml_engine import (
    DATASET_PATHS,
    CLASSIFIERS,
    REGRESSORS,
    load_dataset,
    evaluate_classifier,
    evaluate_regressor,
    run_clustering,
)


class DatasetTests(unittest.TestCase):
    def test_all_fixed_datasets_load(self):
        for key in DATASET_PATHS:
            frame = load_dataset(key)
            self.assertGreater(len(frame), 0, key)
            self.assertGreater(frame.shape[1], 1, key)


class ModelSmokeTests(unittest.TestCase):
    def test_random_forest_classification_metrics_are_valid(self):
        result = evaluate_classifier("Diabetes classification", "Random Forest")
        metrics = result["metrics"]
        for key in ("Accuracy", "Weighted precision", "Macro precision", "Weighted recall", "Macro recall", "Weighted F1-score", "Macro F1-score"):
            self.assertGreaterEqual(metrics[key], 0.0, key)
            self.assertLessEqual(metrics[key], 1.0, key)
        self.assertEqual(result["confusion_matrix"].sum(), metrics["Test rows"])

    def test_random_forest_regression_metrics_are_finite(self):
        result = evaluate_regressor("Insurance regression", "Random Forest Regressor")
        metrics = result["metrics"]
        self.assertTrue(np.isfinite(metrics["R²"]))
        self.assertGreaterEqual(metrics["RMSE"], 0.0)
        self.assertGreaterEqual(metrics["MAE"], 0.0)
        self.assertEqual(len(result["actual"]), metrics["Test rows"])

    def test_clustering_metrics_and_labels(self):
        result = run_clustering(n_clusters=3, include_tsne=False)
        self.assertEqual(set(result["metrics"]["Algorithm"]), {"K-Means", "Agglomerative (Ward)"})
        self.assertEqual(len(result["labels"]["K-Means"]), result["row_count"])
        self.assertTrue(result["metrics"]["Silhouette Score"].between(-1, 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
