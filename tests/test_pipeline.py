import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from main import build_pipeline, prepare_features
import predict


@pytest.fixture
def training_data():
    data = pd.DataFrame({
        "customerID": ["SYN-1", "SYN-2", "SYN-3", "SYN-4"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "One year"],
        "TotalCharges": [" ", "20.5", "40", "60"],
        "Churn": ["Yes", "Yes", "No", "No"],
    })
    return prepare_features(data), data["Churn"]


@pytest.fixture
def unseen_customer():
    return pd.DataFrame({"Contract": ["Two year"], "TotalCharges": [1000.0]})


@pytest.fixture
def fitted_pipeline(training_data):
    features, labels = training_data
    pipeline = build_pipeline(features, LogisticRegression(random_state=42), StandardScaler())
    return pipeline.fit(features, labels)


def test_total_charges_missing_value_is_imputed_before_scaling(training_data, fitted_pipeline):
    features, _ = training_data
    numeric = fitted_pipeline.named_steps["preprocessing"].named_transformers_["numeric"]

    assert np.isnan(features["TotalCharges"].iloc[0])
    assert features["TotalCharges"].iloc[1] == pytest.approx(20.5)
    imputed = numeric.named_steps["imputer"].transform(features[["TotalCharges"]])
    assert imputed[:, 0] == pytest.approx([0.0, 20.5, 40.0, 60.0])


def test_unknown_contract_has_zero_one_hot_columns(fitted_pipeline, unseen_customer):
    encoder = fitted_pipeline.named_steps["preprocessing"].named_transformers_["categorical"]

    assert encoder.transform(unseen_customer[["Contract"]]).tolist() == [[0.0, 0.0]]


def test_preprocessing_keeps_training_statistics_and_categories(fitted_pipeline, unseen_customer):
    preprocessing = fitted_pipeline.named_steps["preprocessing"]
    scaler = preprocessing.named_transformers_["numeric"].named_steps["scaler"]
    encoder = preprocessing.named_transformers_["categorical"]
    expected_mean = np.mean([0.0, 20.5, 40.0, 60.0])
    expected_scale = np.std([0.0, 20.5, 40.0, 60.0])

    transformed = preprocessing.transform(unseen_customer)

    assert scaler.mean_[0] == pytest.approx(expected_mean)
    assert scaler.n_samples_seen_ == 4
    assert encoder.categories_[0].tolist() == ["Month-to-month", "One year"]
    assert transformed[0, -1] == pytest.approx((1000.0 - expected_mean) / expected_scale)


def test_fitted_pipeline_predicts_classes_and_probabilities(fitted_pipeline, unseen_customer):
    classes = fitted_pipeline.predict(unseen_customer)
    probabilities = fitted_pipeline.predict_proba(unseen_customer)

    assert fitted_pipeline.classes_.tolist() == ["No", "Yes"]
    assert classes.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.sum(axis=1) == pytest.approx([1.0])
    assert classes[0] == fitted_pipeline.classes_[probabilities[0].argmax()]


def test_predictions_match_after_joblib_round_trip(fitted_pipeline, tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(predict, "MODEL_PATH", model_path)
    artifact = {"pipeline": fitted_pipeline, "threshold": 0.3}
    customer = {"Contract": "Month-to-month", "TotalCharges": "20.5"}
    before = predict.predict_customer(artifact, customer)

    joblib.dump(artifact, model_path)
    after = predict.predict_customer(predict.load_model(), customer)

    assert after == before
    assert after["threshold"] == 0.3
    assert after["churn"] == ("Yes" if after["probability_churn_yes"] >= 0.3 else "No")
