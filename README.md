# Bank Customer Churn Prediction

This project trains machine learning models to predict customer churn for a bank using MLflow for experiment tracking.

## Project Structure

```
src/
├── train.py          # Main training script with MLflow tracking
├── config.py         # Configuration settings and hyperparameters
├── preprocessing.py  # Data loading and preprocessing functions
├── models.py         # Model creation and training logic
└── evaluation.py     # Metrics calculation and visualization
```

## Dataset

The `Churn_Modelling.csv` dataset contains customer information including credit score, geography, gender, age, tenure, balance, and whether they exited (churned). The preprocessing pipeline balances the classes by downsampling the majority class.

## Setup

Create a virtual environment:

```bash
python -m venv churn_prediction
```

Activate the virtual environment:

```bash
# On Windows
churn_prediction\Scripts\activate

# On Linux/Mac
source churn_prediction/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Start the MLflow tracking server before training:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then run the training script:

```bash
python src/train.py
```

To configure different models and hyperparameters, edit values in `src/config.py`:

- `MODEL_TYPE`: Choose between `"logistic"` or `"random_forest"`
- `LOGISTIC_PARAMS`: Logistic regression hyperparameters
- `RF_PARAMS`: Random forest hyperparameters

## What Gets Logged

Each training run logs:
- Model type and all hyperparameters
- Performance metrics (accuracy, precision, recall, F1 score)
- Trained model with input/output signature
- Column transformer as pickle file
- Training dataset
- Confusion matrix plot

The script also includes detailed logging to track pipeline execution.

## Viewing Results

Open http://127.0.0.1:5000 in your browser to view the MLflow UI. You can compare runs, filter by metrics, and see all logged artifacts.

## Model Registry

After experimenting with different models, register the best ones:

1. Sort runs by F1 score in the UI
2. Click on a run and navigate to the model artifact
3. Click "Register Model" and assign it a name
4. Transition model versions to Staging or Production as needed

## Model Selection Notes

For churn prediction, recall is particularly important since we want to catch as many potential churners as possible. The F1 score gives a good balance between precision and recall for comparing models.

Random Forest generally performs better but is less interpretable. Logistic Regression with regularization (lower C values) can prevent overfitting on smaller datasets.
