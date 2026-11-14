import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_regression.joblib"


def predict_customer(artifact, customer):
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    columns = pipeline.feature_names_in_
    missing = [column for column in columns if column not in customer]
    if missing:
        raise ValueError(f"Missing customer fields: {', '.join(missing)}")

    features = pd.DataFrame([customer], columns=columns)
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    positive_index = list(pipeline.classes_).index("Yes")
    probability = float(pipeline.predict_proba(features)[0, positive_index])
    return {
        "probability_churn_yes": probability,
        "threshold": threshold,
        "churn": "Yes" if probability >= threshold else "No",
    }


def load_model():
    # Load only the artifact produced locally by main.py.
    return joblib.load(MODEL_PATH)


def main():
    parser = argparse.ArgumentParser(description="Predict churn for one customer JSON object.")
    parser.add_argument("customer_json", type=Path)
    args = parser.parse_args()
    try:
        customer = json.loads(args.customer_json.read_text(encoding="utf-8-sig"))
        if not isinstance(customer, dict):
            raise ValueError("Expected one customer as a JSON object.")
        result = predict_customer(load_model(), customer)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
