"""
Configuration settings for the churn prediction model.
"""

# Data paths
DATA_PATH = "dataset/Churn_Modelling.csv"

# MLflow settings
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Churn_Prediction_Experiment"

# Model configuration - change these for different runs
MODEL_TYPE = "logistic"  # "logistic" or "random_forest"

# Logistic Regression params
LOGISTIC_PARAMS = {
    "max_iter": 2000,
    "C": 0.1,
    "solver": "saga"
}

# Random Forest params
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5
}

# Data preprocessing
TEST_SIZE = 0.3
RANDOM_STATE = 1912
RESAMPLE_RANDOM_STATE = 1234

# Feature columns
FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited"
]

CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERICAL_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
