from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


dataset_path = Path(__file__).resolve().parent / "data/synthetic/synthetic_customer_churn.csv"
df = pd.read_csv(dataset_path)
y = df["Churn"]
X = df.drop(columns=["customerID", "Churn"])
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = X_train.select_dtypes(include="str").columns
numeric_features = X_train.select_dtypes(include="number").columns

encoder = OneHotEncoder(handle_unknown="ignore")
imputer = SimpleImputer(strategy="constant", fill_value=0)
scaler = StandardScaler()

X_train_encoded = encoder.fit_transform(X_train[categorical_features]).toarray()
X_train_imputed = imputer.fit_transform(X_train[numeric_features])
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_train_processed = np.hstack([X_train_encoded, X_train_scaled])

X_test_encoded = encoder.transform(X_test[categorical_features]).toarray()
X_test_imputed = imputer.transform(X_test[numeric_features])
X_test_scaled = scaler.transform(X_test_imputed)
X_test_processed = np.hstack([X_test_encoded, X_test_scaled])

print(f"Train rows: {len(y_train)}, test rows: {len(y_test)}")
for name, class_weight in (("Unweighted", None), ("Balanced", "balanced")):
    model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight=class_weight
    )
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label="Yes", zero_division=0
    )
    print(f"\nLogistic regression ({name})")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (Yes): {precision:.4f}")
    print(f"Recall (Yes): {recall:.4f}")
    print(f"F1 (Yes): {f1:.4f}")
    print("Confusion matrix (rows: actual, columns: predicted, order: No, Yes):")
    print(confusion_matrix(y_test, y_pred, labels=["No", "Yes"]))
