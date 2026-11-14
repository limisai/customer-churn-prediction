import pandas as pd
import pytest

from generate_dataset import generate_customers


@pytest.fixture(scope="module")
def customers():
    return generate_customers()


def test_generated_dataset_has_expected_shape_and_schema(customers):
    assert len(customers) == 10_000
    assert customers.columns.tolist() == [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    ]
    assert customers["customerID"].notna().all()
    assert customers["customerID"].is_unique


def test_generator_reproduces_dataset_with_fixed_seed(customers):
    pd.testing.assert_frame_equal(customers, generate_customers())


def test_churn_contains_both_expected_labels(customers):
    assert set(customers["Churn"].unique()) == {"Yes", "No"}
