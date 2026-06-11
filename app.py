"""Streamlit app: interactive hospital diagnosis prediction.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from data_generator import CONDITIONS, DEPARTMENTS, generate_dataset
from models.diagnosis_model import HospitalDiagnosisModel

MODEL_PATH = "models/diagnosis_model.joblib"

st.set_page_config(page_title="Hospital Diagnosis Prediction", page_icon="🏥", layout="centered")


@st.cache_resource
def get_model() -> HospitalDiagnosisModel:
    model = HospitalDiagnosisModel(model_type="random_forest")
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    else:
        # Train on the fly if no artifact is present.
        model.train(generate_dataset())
        model.save_model(MODEL_PATH)
    return model


st.title("🏥 Hospital Diagnosis Prediction")
st.caption(
    "Predicts a patient's likely diagnosis from admission features using a "
    "scikit-learn model trained on a synthetic hospital dataset."
)

model = get_model()

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 90, 50)
    length_of_stay = st.slider("Length of stay (days)", 1, 20, 5)
with col2:
    department = st.selectbox("Department", DEPARTMENTS)
    condition = st.selectbox("Medical condition", CONDITIONS)

if st.button("Predict diagnosis", use_container_width=True):
    patient = {
        "Age": age,
        "Length_of_Stay": length_of_stay,
        "Department": department,
        "Medical_Condition": condition,
    }
    diagnosis, probability = model.predict(patient)
    st.success(f"Predicted diagnosis: **{diagnosis}**  ({probability:.1%} confidence)")

    dist = model.predict_proba(patient)
    st.bar_chart(pd.Series(dist, name="probability"))
