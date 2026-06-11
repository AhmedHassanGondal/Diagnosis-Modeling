"""Hospital diagnosis prediction model.

A self-contained, scikit-learn based classifier that predicts a patient's
diagnosis from a handful of admission features:

    Age, Length_of_Stay, Department, Medical_Condition  ->  Diagnosis

The model wraps a preprocessing + classifier ``Pipeline`` so a single object
handles encoding, scaling and prediction, and can be persisted with joblib.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

NUMERIC_FEATURES = ["Age", "Length_of_Stay"]
CATEGORICAL_FEATURES = ["Department", "Medical_Condition"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Diagnosis"

RANDOM_STATE = 42


class HospitalDiagnosisModel:
    """Predict a patient diagnosis from admission features."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.pipeline: Pipeline | None = None
        self.classes_ = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", self._classifier())]
        )

    def _classifier(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE
            )
        if self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        if self.model_type == "svm":
            return SVC(probability=True, random_state=RANDOM_STATE)
        raise ValueError(
            f"Unknown model_type '{self.model_type}'. Choose from: "
            "random_forest, logistic_regression, svm"
        )

    # ------------------------------------------------------------------ #
    # Training & inference
    # ------------------------------------------------------------------ #
    def train(self, df: pd.DataFrame) -> "HospitalDiagnosisModel":
        """Fit the model on a dataframe containing FEATURES + TARGET."""
        X = df[FEATURES]
        y = df[TARGET]
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_
        return self

    def predict(self, patient_data: dict) -> tuple[str, float]:
        """Predict ``(diagnosis, probability)`` for one patient.

        ``patient_data`` is a dict with keys Age, Length_of_Stay, Department,
        Medical_Condition.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded. Call train() or load_model().")

        row = pd.DataFrame([{key: patient_data[key] for key in FEATURES}])
        proba = self.pipeline.predict_proba(row)[0]
        best = int(proba.argmax())
        diagnosis = str(self.pipeline.named_steps["classifier"].classes_[best])
        return diagnosis, float(proba[best])

    def predict_proba(self, patient_data: dict) -> dict:
        """Return the full probability distribution over diagnoses."""
        if self.pipeline is None:
            raise RuntimeError("Model is not trained or loaded.")
        row = pd.DataFrame([{key: patient_data[key] for key in FEATURES}])
        proba = self.pipeline.predict_proba(row)[0]
        classes = self.pipeline.named_steps["classifier"].classes_
        return {str(c): float(p) for c, p in zip(classes, proba)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_model(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "model_type": self.model_type}, path
        )

    def load_model(self, path: str) -> "HospitalDiagnosisModel":
        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.model_type = artifact.get("model_type", self.model_type)
        self.classes_ = self.pipeline.named_steps["classifier"].classes_
        return self
