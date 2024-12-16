""" Train and evaluate a two-stage random forest classifier for sleep states.

This script trains a two-stage random forest classifier for sleep states using
cross-validation. The first stage of the classifier is a BalancedRandomForest
that predicts whether an event is a wake event or not. The second stage is a
RandomForest that predicts the specific sleep state of the event. The script
loads the dataset, preprocesses the data, and performs cross-validation to
evaluate the model. The evaluation metrics include accuracy, F1 score, 
precision, recall, and AUC-ROC score. The script prints the aggregated metrics
across folds in a tabular format.

Example usage:
    python train_rf.py --data_path data/featured_train_series.csv
"""

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict, List

# Constants
N_SPLITS = 3
RANDOM_STATE = 42


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset, separate features and target, and preprocess the data.

    Parameters:
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Standardized feature matrix and 
        encoded target labels.
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['event', 'series_id'])
    y = df['event']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                stage: str) -> RandomForestClassifier:
    """
    Train a model based on the given stage.

    Parameters:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target labels for training.
        stage (str): Stage of training ('first' or 'second').

    Returns:
        RandomForestClassifier: Trained model.
    """
    if stage == 'first':
        model = BalancedRandomForestClassifier(
            sampling_strategy="all",
            replacement=False,
            bootstrap=True,
            random_state=RANDOM_STATE
        )
        y_train = (y_train != 0).astype(int)
    elif stage == 'second':
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError("Invalid stage specified. Use 'first' or 'second'.")

    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics for predictions.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_onset': f1_score(
            y_true, y_pred, average=None, labels=[1]
            )[0],
        'f1_wakeup': f1_score(
            y_true, y_pred, average=None, labels=[2]
            )[0],
        'precision_onset': precision_score(
            y_true, y_pred, average=None, labels=[1]
            )[0],
        'precision_wakeup': precision_score(
            y_true, y_pred, average=None, labels=[2]
            )[0],
        'recall_onset': recall_score(
            y_true, y_pred, average=None, labels=[1]
            )[0],
        'recall_wakeup': recall_score(
            y_true, y_pred, average=None, labels=[2]
            )[0],
        'auc_roc': roc_auc_score(
            (y_true == 1) | (y_true == 2),
            (y_pred == 1) | (y_pred == 2)
        )
    }


def evaluate_fold(X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   fold: int) -> Dict[str, float]:
    """
    Perform training and evaluation for a single fold.

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Testing feature matrix.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        fold (int): Fold number.

    Returns:
        Dict[str, float]: Metrics computed for the fold.
    """
    print(f"Processing Fold {fold + 1}/{N_SPLITS}")

    # Train first stage model
    first_stage_model = train_model(X_train, y_train, stage='first')
    y_pred_test_bin = first_stage_model.predict(X_test)

    # Filter out non-wake events
    X_train_filtered = X_train[y_train != 0]
    y_train_filtered = y_train[y_train != 0]
    X_test_filtered = X_test[y_pred_test_bin != 0]

    # Train second stage model
    if len(X_test_filtered) > 0:
        second_stage_model = train_model(
            X_train_filtered, y_train_filtered, stage='second')
        y_pred_test_filtered = second_stage_model.predict(X_test_filtered)
        y_pred_combined = np.copy(y_pred_test_bin)
        y_pred_combined[y_pred_test_bin == 1] = y_pred_test_filtered
    else:
        y_pred_combined = y_pred_test_bin

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_combined)

    return metrics


def aggregate_metrics(
    fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across folds.

    Parameters:
        fold_metrics (List[Dict[str, float]]): List of metrics dictionaries 
        for each fold.

    Returns:
        Dict[str, Tuple[float, float]]: Aggregated metrics with mean and 
        standard deviation.
    """
    return {
        key: (
            np.mean([metrics[key] for metrics in fold_metrics]),
            np.std([metrics[key] for metrics in fold_metrics])
        )
        for key in fold_metrics[0].keys()
    }


def print_metrics(aggregated_metrics: Dict[str, Tuple[float, float]]) -> None:
    """
    Print aggregated metrics in a tabular format.

    Parameters:
        aggregated_metrics (Dict[str, Tuple[float, float]]): Aggregated 
        metrics with mean and std.
    """
    print("\n| Metric                     | Mean ± Std                    |")
    print("|----------------------------|-------------------------------|")
    for key, (mean, std) in aggregated_metrics.items():
        print(f"| {key.replace('_', ' ').capitalize():<26} |"
              f"{mean:.4f} ± {std:.4f}                |")


def main(data_path: str) -> None:
    """
    Main script logic for cross-validation training and evaluation.

    Parameters:
        data_path (str): Path to the CSV file containing the dataset.
    """
    # Load data and initialize cross-validation
    X, y = load_data(data_path)
    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        metrics = evaluate_fold(
            X[train_idx], X[test_idx], y[train_idx], y[test_idx], fold)
        fold_metrics.append(metrics)

    # Aggregate and print metrics
    aggregated_metrics = aggregate_metrics(fold_metrics)
    print_metrics(aggregated_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a two-stage classifier.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input CSV file containing the dataset."
    )
    args = parser.parse_args()
    main(args.data_path)
