"""Simple unit tests for the FreshScan AI beginner ML pipeline."""

import numpy as np

from ml_pipeline import (
    check_missing_values,
    clean_dataset,
    load_dataset,
    predict_freshness,
    run_basic_tensorflow_demo,
    train_and_evaluate,
)


def test_load_and_clean_dataset():
    raw_df = load_dataset()
    assert raw_df.shape[0] > 200
    assert "freshness" in raw_df.columns

    missing_df, duplicate_count = check_missing_values(raw_df)
    assert int(missing_df["Missing Values"].sum()) > 0
    assert duplicate_count > 0

    cleaned = clean_dataset(raw_df)
    assert int(cleaned.isnull().sum().sum()) == 0
    assert int(cleaned.duplicated().sum()) == 0


def test_sklearn_models_train_evaluate_and_predict():
    raw_df = load_dataset()

    for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        results = train_and_evaluate(raw_df, model_name=model_name, test_size=0.2)
        assert 0.70 <= results["accuracy"] <= 1.0
        assert results["confusion_matrix"].shape == (2, 2)
        assert int(results["confusion_matrix"].sum()) == results["test_rows"]

        sample_pred = predict_freshness(
            pipeline=results["pipeline"],
            fruit_type="Apple",
            storage_days=3,
            temperature_c=5.0,
            humidity_pct=83.0,
            firmness_score=8.5,
            browning_pct=4.0,
            moisture_loss_pct=2.0,
        )
        assert sample_pred["predicted_label"] in ("Fresh", "Spoiled")
        total_prob = sum(sample_pred["probabilities"].values())
        assert np.isclose(total_prob, 100.0, atol=0.1)


def test_basic_tensorflow_demo():
    raw_df = load_dataset()
    tf_results = run_basic_tensorflow_demo(raw_df, epochs=5, test_size=0.2)
    assert len(tf_results["history"]["loss"]) == 5
    assert len(tf_results["history"]["accuracy"]) == 5
    assert 0.50 <= tf_results["test_accuracy"] <= 1.0
