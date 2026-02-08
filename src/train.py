"""
Main training script for bank customer churn prediction.
Orchestrates data loading, preprocessing, model training, and evaluation with MLflow tracking.
"""



import logging
import os

### Import MLflow
import mlflow

# Import project modules
from preprocessing import load_data, preprocess
from models import train_model
from evaluation import calculate_metrics, log_metrics, create_confusion_matrix
from config import (
    DATA_PATH, MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
    MODEL_TYPE, LOGISTIC_PARAMS, RF_PARAMS
)

# Set user name for MLflow runs
os.environ['LOGNAME'] = 'ahmed reda'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline with MLflow tracking."""
    logger.info("=== Starting Churn Prediction Training Pipeline ===")
    
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    ### Set the experiment name
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"Experiment name: {EXPERIMENT_NAME}")

    # Get model parameters from config
    MODEL_PARAMS = LOGISTIC_PARAMS if MODEL_TYPE == "logistic" else RF_PARAMS
    logger.info(f"Model type: {MODEL_TYPE}")
    logger.info(f"Model parameters: {MODEL_PARAMS}")

    ### Start a new run and leave all the main function code as part of the experiment
    run_name = f"{MODEL_TYPE}_model_{list(MODEL_PARAMS.values())[0]}"
    with mlflow.start_run(run_name=run_name):
        logger.info(f"MLflow run started: {run_name}")
        
        # Load and preprocess data
        df = load_data(DATA_PATH)
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the model parameters
        mlflow.log_param("model_type", MODEL_TYPE)
        for param_name, param_value in MODEL_PARAMS.items():
            mlflow.log_param(param_name, param_value)
        logger.info("Parameters logged to MLflow")

        # Train model (includes MLflow logging inside)
        model = train_model(X_train, y_train, model_type=MODEL_TYPE, **MODEL_PARAMS)

        # Make predictions
        logger.info("Making predictions on test set")
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        metrics = calculate_metrics(y_test, y_pred)
        log_metrics(metrics)

        ### Log tag
        mlflow.set_tag("model_type", MODEL_TYPE)
        mlflow.set_tag("framework", "scikit-learn")
        logger.info("Tags set in MLflow")

        # Create and log confusion matrix artifact
        create_confusion_matrix(y_test, y_pred, model, MODEL_TYPE)
        
        logger.info("=== Training pipeline completed successfully ===")


if __name__ == "__main__":
    main()
