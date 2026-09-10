"""Model preparation and training helpers used by the Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGRESSION_TARGET = "delivery_time_deviation"
CLASSIFICATION_TARGET = "risk_classification"


def prepare_data(path: str) -> pd.DataFrame:
    """Load a training CSV and remove metadata and incomplete rows."""
    data = pd.read_csv(path)
    return _clean_data(data)


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["timestamp"], errors="ignore").dropna().copy()
    missing_targets = {
        target for target in (REGRESSION_TARGET, CLASSIFICATION_TARGET)
        if target not in data.columns
    }
    if missing_targets:
        expected = ", ".join(sorted(missing_targets))
        raise ValueError(f"CSV is missing required target column(s): {expected}")
    return data


def _features(data: pd.DataFrame, target: str) -> pd.DataFrame:
    features = data.drop(columns=[REGRESSION_TARGET, CLASSIFICATION_TARGET])
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        features = pd.get_dummies(features, columns=non_numeric, dtype=float)
    if features.empty:
        raise ValueError("CSV must contain at least one feature column.")
    return features


def train_regression_models(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    data = _clean_data(data)
    features = _features(data, REGRESSION_TARGET)
    target = data[REGRESSION_TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
    }
    fitted: dict[str, Any] = {}
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        fitted[name] = model
        results[name] = {
            "R2 Score": float(r2_score(y_test, prediction)),
            "MAE": float(mean_absolute_error(y_test, prediction)),
        }
    return fitted, results


def train_classification_models(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    data = _clean_data(data)
    features = _features(data, CLASSIFICATION_TARGET)
    target = data[CLASSIFICATION_TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
    }
    fitted: dict[str, Any] = {}
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        fitted[name] = model
        results[name] = {
            "Train Accuracy": float(accuracy_score(y_train, train_prediction)),
            "Test Accuracy": float(accuracy_score(y_test, test_prediction)),
        }
    return fitted, results
