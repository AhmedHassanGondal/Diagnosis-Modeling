"""Train the hospital diagnosis model and persist it.

Generates the dataset if needed, trains a HospitalDiagnosisModel, reports
held-out accuracy, and saves the artifact to models/diagnosis_model.joblib.

Usage:
    python train_model.py --model random_forest
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_generator import generate_dataset
from models.diagnosis_model import FEATURES, TARGET, HospitalDiagnosisModel

DATA_PATH = "data/hospital_patient_dataset.csv"
MODEL_PATH = "models/diagnosis_model.joblib"


def load_or_create_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the diagnosis model.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "logistic_regression", "svm"],
    )
    args = parser.parse_args()

    df = load_or_create_data()
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )

    model = HospitalDiagnosisModel(model_type=args.model)
    model.train(train_df)

    preds = [model.predict(row)[0] for _, row in test_df[FEATURES].iterrows()]
    acc = accuracy_score(test_df[TARGET], preds)
    f1 = f1_score(test_df[TARGET], preds, average="macro")

    model.save_model(MODEL_PATH)
    print(f"Model:     {args.model}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Macro F1:  {f1:.3f}")
    print(f"Saved ->   {MODEL_PATH}")


if __name__ == "__main__":
    main()
