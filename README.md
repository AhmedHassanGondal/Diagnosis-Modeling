# Intelligent Patient Flow and Diagnosis Modeling in Smart Hospitals

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

A project exploring stochastic processes and machine learning for modeling patient
flow and diagnosis prediction in a hospital setting.

> **Status — work in progress.** This repository currently contains a minimal
> Streamlit starter app and test scaffolding. The full modeling system described
> under [Planned scope](#planned-scope-not-yet-implemented) is **not yet
> implemented here**. The sections below describe only what is actually present.

## Current contents

```
.
├── simple_app.py        # Minimal Streamlit app (smoke test that Streamlit works)
├── test_imports.py      # Prints versions of the core dependencies, checks imports
├── test_diagnosis.py    # Scaffold test for a diagnosis model (see note below)
├── requirements.txt     # Project dependencies
└── README.md
```

> **Note:** `test_diagnosis.py` imports `models.diagnosis_model.HospitalDiagnosisModel`
> and loads `models/diagnosis_model.joblib`. Neither the `models/` package nor the
> saved model file exists in this repository yet, so this test will fail until the
> diagnosis model is added.

## Installation and setup

1. Clone the repository:
   ```
   git clone https://github.com/AhmedHassanGondal/Diagnosis-Modeling.git
   cd Diagnosis-Modeling
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the starter Streamlit app:
   ```
   streamlit run simple_app.py
   ```

4. (Optional) Verify your environment has the dependencies installed:
   ```
   python test_imports.py
   ```

## Planned scope (not yet implemented)

The intended system combines stochastic processes, machine learning, and
interactive visualizations to model and analyze patient flow, waiting times,
resource utilization, and diagnosis prediction. None of the modules below are
present in this repository yet — they describe the design goal:

- **Patient Flow Simulation** — Markov chains for department transitions
  (ER, Ward, ICU, Discharged).
- **Hidden Markov Models** — infer unobservable patient health states
  (Stable, Deteriorating, Critical) from observed symptoms/vitals.
- **Poisson Arrival Process** — model patient arrivals with time-varying rates.
- **Hospital Queuing Theory** — model waiting/service times per department
  (M/M/c, M/G/c, priority queues).
- **Bayesian Networks** — probabilistic relationships between symptoms,
  diseases, and treatments.
- **Diagnosis Prediction** — Random Forest, Logistic Regression, and SVM
  classifiers over patient features.
- **3D Interactive Visualizations** — immersive views of patient flow and models.
- **Interactive Dashboard** — a unified Streamlit interface for all of the above.

### Intended dataset features

A synthetic hospital-patient dataset is planned with demographics (Age, Gender,
Smoking), hospital information (Department, Length of Stay, Medical Condition,
Diagnosis), and symptom/vital-sign indicators (Chest Pain, Shortness of Breath,
Fatigue, Cough, Excessive Thirst, ECG Abnormal).

## Key technologies

- **Python** — core language
- **Streamlit** — interactive web app framework
- **NumPy / Pandas** — data manipulation and analysis
- **Matplotlib / Seaborn** — static visualization
- **Plotly** — interactive 2D/3D visualization
- **SciPy** — scientific computing and statistical distributions
- **Scikit-learn** — machine learning algorithms
- **hmmlearn** — Hidden Markov Model implementation
- **NetworkX** — graph-based modeling for Bayesian networks

(See `requirements.txt` for the full list.)

## License

This project is intended to be licensed under the MIT License. A `LICENSE` file
is not yet included in the repository.

## Acknowledgments

- Developed as part of a stochastic processes and healthcare analytics course.
- Inspired by real-world hospital operations and patient flow modeling.
