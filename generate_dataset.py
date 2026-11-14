from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent / "data/synthetic/synthetic_customer_churn.csv"
CUSTOMER_COUNT = 10_000
SEED = 42


def generate_customers():
    rng = np.random.default_rng(SEED)
    n = CUSTOMER_COUNT

    def sample(probability):
        return rng.random(n) < probability

    def yes_no(mask):
        return np.where(mask, "Yes", "No")

    tenure = rng.integers(0, 73, size=n)
    senior = sample(0.17)
    partner = sample(0.46)
    dependents = sample(np.where(partner, 0.46, 0.18))

    contract_draw = rng.random(n)
    two_year_probability = np.where(tenure >= 24, 0.10 + tenure / 240, 0.04)
    one_year_probability = np.where(tenure >= 12, 0.23, 0.10)
    contract = np.select(
        [
            contract_draw < two_year_probability,
            contract_draw < two_year_probability + one_year_probability,
        ],
        ["Two year", "One year"],
        default="Month-to-month",
    )
    monthly_contract = contract == "Month-to-month"

    phone = sample(0.90)
    multiple_lines = phone & sample(0.26 + 0.003 * tenure)
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.36, 0.45, 0.19])
    has_internet = internet != "No"
    fiber = internet == "Fiber optic"
    security = has_internet & sample(np.where(monthly_contract, 0.22, 0.46))
    backup = has_internet & sample(0.27 + 0.003 * tenure)
    protection = has_internet & sample(0.34)
    support = has_internet & sample(np.where(monthly_contract, 0.20, 0.44))
    streaming_tv = has_internet & sample(np.where(fiber, 0.64, 0.33))
    streaming_movies = has_internet & sample(np.where(streaming_tv, 0.68, 0.30))
    paperless = sample(np.where(has_internet, 0.68, 0.34))
    payment = rng.choice(["Bank transfer (automatic)", "Credit card (automatic)", "Mailed check"], size=n)
    payment = np.where(
        sample(np.where(monthly_contract & paperless, 0.48, 0.13)),
        "Electronic check",
        payment,
    )

    monthly_charges = 18.0 + 9 * phone + 7 * multiple_lines
    monthly_charges += np.select([fiber, internet == "DSL"], [49, 29], default=0)
    monthly_charges += 5 * security + 5 * backup + 5 * protection + 6 * support
    monthly_charges += 9 * streaming_tv + 9 * streaming_movies
    monthly_charges -= np.select([contract == "Two year", contract == "One year"], [8, 4], default=0)
    monthly_charges = np.round(np.maximum(10, monthly_charges + rng.normal(0, 3, n)), 2)
    # Past bills can differ from the current bill because of plan changes and discounts.
    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.85, 1.15, n), 2)

    # Sample churn after building the features. Independent noise represents unobserved factors.
    log_odds = (
        -1.25 + 1.0 * monthly_contract - 0.025 * tenure
        + 0.018 * (monthly_charges - 65) + 0.36 * fiber
        - 0.48 * support - 0.32 * security
        + 0.48 * (payment == "Electronic check") + 0.24 * senior
        + 0.50 * (monthly_contract & fiber) + rng.normal(0, 0.55, n)
    )
    churn = sample(1 / (1 + np.exp(-log_odds)))

    def internet_service(mask):
        return np.where(has_internet, yes_no(mask), "No internet service")

    return pd.DataFrame({
        "customerID": [f"SYN-{i:05d}" for i in range(1, n + 1)],
        "gender": rng.choice(["Female", "Male"], size=n),
        "SeniorCitizen": senior.astype(int),
        "Partner": yes_no(partner),
        "Dependents": yes_no(dependents),
        "tenure": tenure,
        "PhoneService": yes_no(phone),
        "MultipleLines": np.where(phone, yes_no(multiple_lines), "No phone service"),
        "InternetService": internet,
        "OnlineSecurity": internet_service(security),
        "OnlineBackup": internet_service(backup),
        "DeviceProtection": internet_service(protection),
        "TechSupport": internet_service(support),
        "StreamingTV": internet_service(streaming_tv),
        "StreamingMovies": internet_service(streaming_movies),
        "Contract": contract,
        "PaperlessBilling": yes_no(paperless),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": yes_no(churn),
    })


if __name__ == "__main__":
    customers = generate_customers()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(DATASET_PATH, index=False, float_format="%.2f", lineterminator="\n")
    print(f"Wrote {len(customers)} synthetic customers to {DATASET_PATH}")
