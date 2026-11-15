"""Predict a student's performance category with the saved model."""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "student_performance_model.joblib"


def ask_number(prompt):
    while True:
        value = input(prompt).strip()
        try:
            return float(value)
        except ValueError:
            print("Number enter karo, jaise 6 ya 7.5.")


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file nahi mili: {MODEL_FILE}. Pehle student_performance_model.py run karo."
        )

    student = {
        "Hours Studied": ask_number("Hours Studied: "),
        "Sleep Hours": ask_number("Sleep Hours: "),
        "Previous Scores": ask_number("Previous Scores: "),
        "Sample Question Papers Practiced": ask_number(
            "Sample Question Papers Practiced: "
        ),
    }
    activity = input("Extracurricular Activities (Yes/No): ").strip().casefold()
    if activity not in {"yes", "no"}:
        raise ValueError("Extracurricular Activities mein Yes ya No enter karo.")
    student["Extracurricular Activities"] = "Yes" if activity == "yes" else "No"

    model = joblib.load(MODEL_FILE)
    student_data = pd.DataFrame([student])
    category = model.predict(student_data)[0]
    print(f"\nPredicted Performance Category: {category}")

    probabilities = model.predict_proba(student_data)[0]
    classes = model.named_steps["classifier"].classes_
    print("Class probabilities:")
    for label, probability in zip(classes, probabilities):
        print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()

