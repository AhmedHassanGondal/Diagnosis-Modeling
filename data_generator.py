"""Generate a synthetic hospital patient dataset.

The relationships are deterministic-with-noise so a classifier can learn them:
the diagnosis depends mostly on the presenting Medical_Condition, modulated by
Age, Department and Length_of_Stay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42

DEPARTMENTS = ["ER", "Ward", "ICU"]
CONDITIONS = ["Cardiac", "Neuro", "Injury"]
DIAGNOSES = ["Condition A", "Condition B", "Condition C"]


def generate_dataset(n: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 90, size=n)
    length_of_stay = rng.integers(1, 21, size=n)
    department = rng.choice(DEPARTMENTS, size=n, p=[0.5, 0.3, 0.2])
    condition = rng.choice(CONDITIONS, size=n, p=[0.4, 0.3, 0.3])

    # Base diagnosis tendency from the medical condition.
    base = {"Cardiac": [0.7, 0.2, 0.1], "Neuro": [0.15, 0.7, 0.15], "Injury": [0.1, 0.2, 0.7]}

    diagnosis = []
    for i in range(n):
        probs = np.array(base[condition[i]], dtype=float)
        # Older patients & ICU stays shift weight toward "Condition A" (severe).
        if age[i] > 65:
            probs[0] += 0.15
        if department[i] == "ICU" or length_of_stay[i] > 12:
            probs[0] += 0.10
        probs = probs / probs.sum()
        diagnosis.append(rng.choice(DIAGNOSES, p=probs))

    return pd.DataFrame(
        {
            "Age": age,
            "Length_of_Stay": length_of_stay,
            "Department": department,
            "Medical_Condition": condition,
            "Diagnosis": diagnosis,
        }
    )


if __name__ == "__main__":  # pragma: no cover
    import os

    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/hospital_patient_dataset.csv", index=False)
    print(f"Wrote {len(df)} rows to data/hospital_patient_dataset.csv")
    print(df.head())
