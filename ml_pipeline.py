"""
FreshScan AI — Beginner-Friendly Machine Learning Pipeline
===========================================================
This module keeps all data loading, cleaning, Scikit-learn preprocessing,
model training, evaluation, and optional introductory TensorFlow/Keras
steps in simple, easy-to-read Python functions.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Default path to our educational sample dataset
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "fruit_freshness_dataset.csv"

# Separate feature columns by data type for preprocessing
NUMERIC_FEATURES = [
    "storage_days",
    "temperature_c",
    "humidity_pct",
    "firmness_score",
    "browning_pct",
    "moisture_loss_pct",
]
CATEGORICAL_FEATURES = ["fruit_type"]
TARGET_COLUMN = "freshness"


def load_dataset(csv_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Step 1: Load the CSV dataset into a Pandas DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    return pd.read_csv(path)


def check_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Step 3: Count missing values per column and total duplicate rows."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    summary_df = pd.DataFrame(
        {
            "Column": missing_counts.index,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": missing_counts.values,
            "Missing (%)": missing_pct.values,
        }
    )
    duplicate_count = int(df.duplicated().sum())
    return summary_df, duplicate_count


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Basic data cleaning before exploration and plotting.
    - Drop exact duplicate rows.
    - Fill missing numeric values with each column's median.
    - Fill missing text values with the most common category.
    """
    cleaned = df.drop_duplicates().copy()

    for col in NUMERIC_FEATURES:
        if col in cleaned.columns and cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in cleaned.columns and cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return cleaned


def get_classifier(model_name: str, random_state: int = 42):
    """Step 7: Choose an understandable Scikit-learn classification model."""
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=random_state)
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model selection: {model_name}")


def build_pipeline(model_name: str = "Random Forest", random_state: int = 42) -> Pipeline:
    """
    Step 8: Create a clean Scikit-learn Pipeline.
    Why use a Pipeline?
    It bundles preprocessing (imputing, scaling, encoding) and the classifier
    into one object so training and new predictions follow the exact same rules
    without data leakage.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    classifier = get_classifier(model_name, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract simple feature importance (Tree/Forest) or coefficient magnitude (LogReg)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.replace("num__", "").replace("cat__fruit_type_", "Fruit: ")
        for name in feature_names
    ]

    if hasattr(classifier, "feature_importances_"):
        scores = classifier.feature_importances_
        metric_label = "Importance Score"
    elif hasattr(classifier, "coef_"):
        scores = np.abs(classifier.coef_[0])
        metric_label = "Absolute Weight"
    else:
        return pd.DataFrame()

    importance_df = (
        pd.DataFrame({"Feature": clean_names, metric_label: np.round(scores, 4)})
        .sort_values(by=metric_label, ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def train_and_evaluate(
    df: pd.DataFrame,
    model_name: str = "Random Forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Steps 6–9:
    1. Remove duplicate rows (note: Pipeline handles missing values safely inside train/test).
    2. Split data into training and testing sets using train_test_split.
    3. Fit the Scikit-learn Pipeline on training data.
    4. Evaluate real predictions on unseen test data using accuracy and confusion matrix.
    """
    working_df = df.drop_duplicates().dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    X = working_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = working_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(model_name=model_name, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = ["Fresh", "Spoiled"]

    accuracy = float(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "accuracy": accuracy,
        "labels": labels,
        "confusion_matrix": conf_matrix,
        "classification_report": report_df,
        "feature_importance": _extract_feature_importance(pipeline),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_clean_rows": len(working_df),
    }


def predict_freshness(
    pipeline: Pipeline,
    fruit_type: str,
    storage_days: float,
    temperature_c: float,
    humidity_pct: float,
    firmness_score: float,
    browning_pct: float,
    moisture_loss_pct: float,
) -> Dict[str, Any]:
    """Use the trained Scikit-learn Pipeline to predict Fresh vs. Spoiled for one sample."""
    input_row = pd.DataFrame(
        [
            {
                "fruit_type": fruit_type,
                "storage_days": storage_days,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "firmness_score": firmness_score,
                "browning_pct": browning_pct,
                "moisture_loss_pct": moisture_loss_pct,
            }
        ]
    )

    predicted_label = str(pipeline.predict(input_row)[0])
    probabilities = pipeline.predict_proba(input_row)[0]
    class_names = list(pipeline.classes_)
    prob_map = {
        str(cls): round(float(prob) * 100.0, 2)
        for cls, prob in zip(class_names, probabilities)
    }

    return {
        "predicted_label": predicted_label,
        "confidence_pct": prob_map.get(predicted_label, 0.0),
        "probabilities": prob_map,
    }


def run_basic_tensorflow_demo(
    df: pd.DataFrame,
    epochs: int = 15,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optional Beginner TensorFlow/Keras Demonstration:
    Trains a very small neural network (Input -> Hidden 16 -> Hidden 8 -> Output 1)
    on the same cleaned CSV dataset so students can observe Epoch, Loss, and Accuracy.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(random_state)

    cleaned = clean_dataset(df)
    X = pd.get_dummies(cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES], drop_first=False)
    y = (cleaned[TARGET_COLUMN] == "Spoiled").astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32),
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simple 3-layer Neural Network using Keras Sequential API
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
    )

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    return {
        "tf_version": tf.__version__,
        "num_input_features": int(X_train_scaled.shape[1]),
        "epochs": epochs,
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
