"""Evaluate a student performance category model on held-out data."""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "StudentPerformance.csv"
FEATURES = [
    "Hours Studied",
    "Sleep Hours",
    "Previous Scores",
    "Sample Question Papers Practiced",
    "Extracurricular Activities",
]
TARGET = "Performance Index"
CATEGORY_ORDER = ["Low", "Medium", "High"]


def main():
    data = pd.read_csv(DATA_FILE)

    # Remove exact duplicate rows, matching the existing notebook's cleaning step.
    data = data.drop_duplicates().copy()
    data["PerformanceCategory"] = pd.cut(
        data[TARGET],
        bins=[float("-inf"), 50, 75, float("inf")],
        labels=CATEGORY_ORDER,
        right=False,
    )

    X = data[FEATURES]
    y = data["PerformanceCategory"]

    numeric_features = [
        "Hours Studied",
        "Sleep Hours",
        "Previous Scores",
        "Sample Question Papers Practiced",
    ]
    categorical_features = ["Extracurricular Activities"]
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Rows used after removing duplicates: {len(data)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

    models = {
        "Baseline": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Balanced": DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=42
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, classifier in models.items():
        model = Pipeline(
            steps=[("preprocess", preprocess), ("classifier", classifier)]
        )
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        )
        print(f"\n{name} model")
        print(
            "5-fold CV accuracy: "
            f"{cv_scores['test_accuracy'].mean():.3f} "
            f"(+/- {cv_scores['test_accuracy'].std():.3f})"
        )
        print(
            "5-fold CV macro F1: "
            f"{cv_scores['test_macro_f1'].mean():.3f} "
            f"(+/- {cv_scores['test_macro_f1'].std():.3f})"
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
        print("Classification report (held-out test set):")
        print(classification_report(y_test, predictions, labels=CATEGORY_ORDER, zero_division=0))
        print("Confusion matrix (rows=actual, columns=predicted; Low, Medium, High):")
        print(confusion_matrix(y_test, predictions, labels=CATEGORY_ORDER))

    # Refit the selected model on all labeled rows for future student predictions.
    final_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=3, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    final_model.fit(X, y)
    model_path = PROJECT_DIR / "student_performance_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"\nFinal balanced model saved to: {model_path}")


if __name__ == "__main__":
    main()

