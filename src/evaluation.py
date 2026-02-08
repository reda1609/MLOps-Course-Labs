"""
Model evaluation and metrics functions.
"""

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import mlflow

logger = logging.getLogger(__name__)


def calculate_metrics(y_test, y_pred):
    """Calculate classification metrics."""
    logger.info("Calculating metrics")
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    
    return metrics


def log_metrics(metrics):
    """Log all metrics to MLflow."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    logger.info("Metrics logged to MLflow")


def create_confusion_matrix(y_test, y_pred, model, model_type):
    """
    Create and save confusion matrix plot.
    
    Returns path to saved figure.
    """
    logger.info("Creating confusion matrix")
    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    
    artifact_name = f"confusion_matrix_{model_type}.png"
    plt.savefig(artifact_name)
    logger.info(f"Confusion matrix saved to {artifact_name}")
    
    # Log as artifact
    mlflow.log_artifact(artifact_name)
    logger.info("Confusion matrix logged to MLflow")
    
    plt.show()
    
    return artifact_name
