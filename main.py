from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict import MODEL_PATH


dataset_path = Path(__file__).resolve().parent / "data/synthetic/synthetic_customer_churn.csv"
plot_path = dataset_path.parents[2] / "docs" / "threshold_tradeoff.png"


def prepare_features(data):
    features = data.drop(columns=["customerID", "Churn"])
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    return features


def build_pipeline(X_train, classifier, scaler):
    categorical_features = X_train.select_dtypes(include="str").columns
    numeric_features = X_train.select_dtypes(include="number").columns
    numeric_preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", scaler),
    ])
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("numeric", numeric_preprocessing, numeric_features),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])



def choose_threshold(model, X_train, y_train, folds):
    positive_index = list(np.unique(y_train)).index("Yes")
    probabilities = cross_val_predict(
        model, X_train, y_train, cv=folds, method="predict_proba"
    )[:, positive_index]
    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)
    scores = []
    print("Train OOF threshold comparison (Yes):")
    print("Threshold  Precision  Recall      F1")
    for threshold in thresholds:
        predictions = np.where(probabilities >= threshold, "Yes", "No")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_train, predictions, average="binary", pos_label="Yes", zero_division=0
        )
        scores.append((threshold, precision, recall, f1))
        print(f"{threshold:>9.1f}  {precision:>9.4f}  {recall:>6.4f}  {f1:>6.4f}")

    # Candidates are ordered ascending, so an exact tie favors the lower threshold.
    selected = max(scores, key=lambda row: row[3])[0]
    figure = Figure(figsize=(7, 4), layout="constrained")
    axes = figure.subplots()
    axes.plot(thresholds, [row[1] for row in scores], "o-", label="Precision")
    axes.plot(thresholds, [row[2] for row in scores], "s-", label="Recall")
    axes.axvline(selected, color="gray", linestyle="--", label=f"Selected: {selected:.1f}")
    axes.set(title="Churn threshold comparison on train (OOF)", xlabel="Threshold",
             ylabel="Score for Yes", xticks=thresholds, ylim=(0, 1))
    axes.grid(alpha=0.25)
    axes.legend()
    plot_path.parent.mkdir(exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    print(f"Selected threshold: {selected:.1f}")
    return selected


def report_test_metrics(y_test, predictions, probabilities):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", pos_label="Yes", zero_division=0
    )
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision (Yes): {precision:.4f}")
    print(f"Recall (Yes): {recall:.4f}")
    print(f"F1 (Yes): {f1:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test == 'Yes', probabilities):.4f}")
    print("Confusion matrix (rows: actual, columns: predicted, order: No, Yes):")
    print(confusion_matrix(y_test, predictions, labels=["No", "Yes"]))


def report_feature_importance(model, X_train, y_train, folds, fold_models, threshold):
    def threshold_f1(estimator, features, labels):
        positive_index = list(estimator.classes_).index("Yes")
        probabilities = estimator.predict_proba(features)[:, positive_index]
        predictions = np.where(probabilities >= threshold, "Yes", "No")
        return f1_score(labels, predictions, pos_label="Yes", zero_division=0)

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    coefficients = pd.Series(classifier.coef_[0], index=feature_names)
    top_coefficients = coefficients.loc[coefficients.abs().nlargest(15).index]
    print("\nTop 15 coefficients by absolute value:")
    print(top_coefficients.to_string(float_format="{:+.4f}".format))

    fold_importances = []
    for estimator, (_, validation) in zip(fold_models, folds.split(X_train, y_train), strict=True):
        result = permutation_importance(
            estimator, X_train.iloc[validation], y_train.iloc[validation],
            scoring=threshold_f1, n_repeats=10, random_state=42,
        )
        fold_importances.append(result.importances_mean)
    importance = pd.Series(np.mean(fold_importances, axis=0), index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    print("\nPermutation importance (mean validation F1 decrease):")
    print(importance.to_string(float_format="{:+.4f}".format))

    for values, filename, title, xlabel in (
        (top_coefficients, "churn_coefficients.png", "Top 15 logistic regression coefficients",
         "Coefficient for Yes (log-odds)"),
        (importance, "churn_permutation_importance.png", "Permutation importance on train folds",
         f"Mean decrease in F1 for Yes (threshold {threshold:.1f})"),
    ):
        figure = Figure(figsize=(9, 6), layout="constrained")
        axes = figure.subplots()
        colors = np.where(values >= 0, "#176c9e", "#b64836")
        axes.barh(values.index, values, color=colors)
        axes.invert_yaxis()
        axes.axvline(0, color="black", linewidth=0.8)
        axes.set(title=title, xlabel=xlabel)
        figure.savefig(plot_path.with_name(filename), dpi=150)


def main():
    df = pd.read_csv(dataset_path)
    y = df["Churn"]
    X = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = (
        ("Logistic regression (Unweighted)", LogisticRegression(max_iter=1000, random_state=42), StandardScaler()),
        ("Logistic regression (Balanced)", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), StandardScaler()),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), "passthrough"),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "F1": make_scorer(f1_score, pos_label="Yes", zero_division=0),
        "Precision": make_scorer(precision_score, pos_label="Yes", zero_division=0),
        "Recall": make_scorer(recall_score, pos_label="Yes", zero_division=0),
    }

    print(f"Train rows: {len(y_train)}, test rows: {len(y_test)}")
    for index, (name, classifier, scaler) in enumerate(models):
        model = build_pipeline(X_train, classifier, scaler)
        cv_results = cross_validate(
            model, X_train, y_train, cv=folds, scoring=scoring, error_score="raise",
            return_estimator=index == 0,
        )
        print(f"\n{name}")
        print("5-fold stratified CV on train (mean +/- SD):")
        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            print(f"{metric} (Yes): {scores.mean():.4f} +/- {scores.std():.4f}")

        selected_threshold = choose_threshold(model, X_train, y_train, folds) if index == 0 else None
        model.fit(X_train, y_train)
        if selected_threshold is not None:
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump({"pipeline": model, "threshold": selected_threshold}, MODEL_PATH)
            report_feature_importance(
                model, X_train, y_train, folds, cv_results["estimator"], selected_threshold
            )
        y_pred = model.predict(X_test)
        positive_index = list(model.classes_).index("Yes")
        churn_probability = model.predict_proba(X_test)[:, positive_index]

        print("Held-out test:")
        report_test_metrics(y_test, y_pred, churn_probability)
        if selected_threshold is not None:
            threshold_predictions = np.where(churn_probability >= selected_threshold, "Yes", "No")
            print(f"Held-out test (selected threshold {selected_threshold:.1f}):")
            report_test_metrics(y_test, threshold_predictions, churn_probability)


if __name__ == "__main__":
    main()
