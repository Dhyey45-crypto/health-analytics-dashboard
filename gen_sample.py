import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

n = 200

names = [f"Patient_{i}" for i in range(1, n+1)]
genders = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])
ages = np.random.randint(1, 90, size=n)

diseases = np.random.choice(
    ["Diabetes", "Hypertension", "Asthma", "Cardiac Disorder", "Obesity",
     "Thyroid Disorder", "Arthritis", "None"],
    size=n,
    p=[0.14, 0.16, 0.10, 0.10, 0.10, 0.08, 0.10, 0.22]
)

# BMI generation loosely correlated with disease
bmi = np.round(np.random.normal(loc=25, scale=5, size=n), 1)
bmi = np.clip(bmi, 14, 48)

systolic = np.round(np.random.normal(loc=120, scale=15, size=n)).astype(int)
diastolic = np.round(np.random.normal(loc=80, scale=10, size=n)).astype(int)
systolic = np.clip(systolic, 90, 190)
diastolic = np.clip(diastolic, 55, 120)

feedback_options = [
    "The doctor was very helpful and caring, I feel much better now.",
    "Terrible experience, long waiting time and rude staff.",
    "Average service, nothing special but okay treatment.",
    "Excellent care, the nurses were kind and attentive.",
    "I am not satisfied with the treatment plan given.",
    "Great hospital, quick diagnosis and effective medication.",
    "The staff was okay but the facility needs improvement.",
    "Wonderful experience overall, highly recommend this clinic.",
    "Disappointed with the delay in reports and follow-up.",
    "Good doctors but the billing process was confusing."
]
feedback = np.random.choice(feedback_options, size=n)

admission_dates = pd.date_range("2024-01-01", periods=365, freq="D")
admission_date = np.random.choice(admission_dates, size=n)

df = pd.DataFrame({
    "PatientID": range(1, n+1),
    "Name": names,
    "Age": ages,
    "Gender": genders,
    "Disease": diseases,
    "BMI": bmi,
    "BP_Systolic": systolic,
    "BP_Diastolic": diastolic,
    "AdmissionDate": pd.to_datetime(admission_date).strftime("%Y-%m-%d"),
    "Feedback": feedback
})

df.to_csv("sample_data/patients_sample.csv", index=False)
print(df.head())
print("Saved:", len(df), "rows")
