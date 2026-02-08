"""
Data preprocessing functions for churn prediction.
"""

import pandas as pd
import pickle
import logging
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow

from config import (
    FEATURE_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    TEST_SIZE, RANDOM_STATE, RESAMPLE_RANDOM_STATE
)

logger = logging.getLogger(__name__)


def load_data(data_path):
    """Load the dataset from CSV."""
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        raise


def rebalance(data):
    """
    Resample data to keep balance between target classes.
    Downsample majority class to match minority class size.
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    
    logger.info(f"Rebalancing data - Majority: {len(churn_maj)}, Minority: {len(churn_min)}")
    
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), 
        replace=False, random_state=RESAMPLE_RANDOM_STATE
    )
    
    balanced_data = pd.concat([churn_maj_downsample, churn_min])
    logger.info(f"Balanced dataset size: {len(balanced_data)}")
    
    return balanced_data


def preprocess(df):
    """
    Preprocess and split data into training and test sets.
    
    Returns:
        col_transf: ColumnTransformer with scalers and encoders
        X_train, X_test: Transformed feature DataFrames
        y_train, y_test: Target Series
    """
    logger.info("Starting data preprocessing")
    
    data = df.loc[:, FEATURE_COLS]
    data_bal = rebalance(data=data)
    
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train/test split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    col_transf = make_column_transformer(
        (StandardScaler(), NUMERICAL_COLS),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_COLS),
        remainder="passthrough",
    )
    
    logger.info("Fitting transformer on training data")
    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())
    
    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())
    
    # Save transformer to pickle file
    transformer_path = "column_transformer.pkl"
    with open(transformer_path, "wb") as f:
        pickle.dump(col_transf, f)
    logger.info(f"Transformer saved to {transformer_path}")
    
    # Log the pickle file as an artifact
    mlflow.log_artifact(transformer_path)
    
    return col_transf, X_train, X_test, y_train, y_test
