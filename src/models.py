"""
Model training functions.
"""

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow

logger = logging.getLogger(__name__)


def create_model(model_type, **model_params):
    """Create a model instance based on type and parameters."""
    if model_type == "random_forest":
        logger.info(f"Creating Random Forest model with params: {model_params}")
        return RandomForestClassifier(**model_params, random_state=42)
    else:  # logistic
        logger.info(f"Creating Logistic Regression model with params: {model_params}")
        return LogisticRegression(**model_params)


def train_model(X_train, y_train, model_type="logistic", **model_params):
    """
    Train a classification model and log it to MLflow.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model ("logistic" or "random_forest")
        **model_params: Model hyperparameters
    
    Returns:
        Trained model
    """
    logger.info(f"Training {model_type} model")
    
    model = create_model(model_type, **model_params)
    model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    
    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    
    # Log model
    mlflow.sklearn.log_model(model, f"{model_type}_model", signature=signature)
    logger.info("Model logged to MLflow")
    
    ### Log the data
    mlflow.log_input(mlflow.data.from_pandas(X_train), context="training")
    logger.info("Training data logged to MLflow")
    
    return model
