"""
data_processor.py
-------------------
Helper functions for cleaning, validating and enriching patient data
used by the Health Analytics Dashboard.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob

REQUIRED_COLUMNS = [
    "PatientID", "Name", "Age", "Gender", "Disease",
    "BMI", "BP_Systolic", "BP_Diastolic"
]


def validate_columns(df: pd.DataFrame):
    """Check whether the uploaded dataframe has the minimum required columns.

    Returns (is_valid, missing_columns)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop empty rows, coerce numeric types, strip strings."""
    df = df.copy()

    # Strip whitespace from string/object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Coerce numeric columns
    for col in ["Age", "BMI", "BP_Systolic", "BP_Diastolic"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical numeric fields
    df = df.dropna(subset=[c for c in ["Age", "BMI"] if c in df.columns])

    return df


def bmi_category(bmi: float) -> str:
    """Classify BMI into standard WHO categories."""
    if pd.isna(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def bp_category(systolic: float, diastolic: float) -> str:
    """Classify blood pressure into standard categories (AHA guidelines, simplified)."""
    if pd.isna(systolic) or pd.isna(diastolic):
        return "Unknown"
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Hypertension Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2"
    else:
        return "Unknown"


def age_group(age: float) -> str:
    """Bucket ages into readable groups."""
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age <= 12:
        return "Child (0-12)"
    elif age <= 19:
        return "Teen (13-19)"
    elif age <= 35:
        return "Young Adult (20-35)"
    elif age <= 60:
        return "Adult (36-60)"
    else:
        return "Senior (60+)"


def sentiment_polarity(text: str) -> float:
    """Return TextBlob sentiment polarity score (-1 to 1) for a text string."""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return TextBlob(text).sentiment.polarity


def sentiment_label(polarity: float) -> str:
    """Convert a polarity score into a human readable label."""
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (BMI category, BP category, Age group, Sentiment)."""
    df = df.copy()

    if "BMI" in df.columns:
        df["BMI_Category"] = df["BMI"].apply(bmi_category)

    if "BP_Systolic" in df.columns and "BP_Diastolic" in df.columns:
        df["BP_Category"] = df.apply(
            lambda r: bp_category(r["BP_Systolic"], r["BP_Diastolic"]), axis=1
        )

    if "Age" in df.columns:
        df["Age_Group"] = df["Age"].apply(age_group)

    if "Feedback" in df.columns:
        df["Sentiment_Polarity"] = df["Feedback"].apply(sentiment_polarity)
        df["Sentiment_Label"] = df["Sentiment_Polarity"].apply(sentiment_label)

    return df
