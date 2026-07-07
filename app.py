"""
Health Analytics Dashboard
--------------------------
A Streamlit + Plotly + TextBlob powered dashboard for exploring patient
health records: demographics, disease burden, BMI, blood pressure and
patient-feedback sentiment.

Run with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import nltk

@st.cache_resource
def ensure_nltk_data():
    for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "brown"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_data()

from utils.data_processor import (
    validate_columns,
    clean_data,
    enrich_dataframe,
)

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#2563EB"
ACCENT_COLOR = "#0EA5E9"

CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px 8px 16px;
    }
    div[data-testid="stMetricLabel"] {font-weight: 600; color:#334155;}
    h1, h2, h3 {color:#0F172A;}
    .section-divider {
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E2E8F0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar — Data Upload
# --------------------------------------------------------------------------
st.sidebar.title("🏥 Health Analytics")
st.sidebar.markdown("Upload patient data or use the bundled sample dataset.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Patient Data (CSV)", type=["csv"], help="CSV must include columns like "
    "PatientID, Name, Age, Gender, Disease, BMI, BP_Systolic, BP_Diastolic"
)

use_sample = st.sidebar.checkbox("Use sample dataset", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Required columns: PatientID, Name, Age, Gender, Disease, BMI, "
    "BP_Systolic, BP_Diastolic.\nOptional: Feedback (for sentiment analysis), "
    "AdmissionDate."
)

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
@st.cache_data
def load_sample():
    return pd.read_csv("sample_data/patients_sample.csv")


raw_df = None
if uploaded_file is not None and not use_sample:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
elif use_sample:
    raw_df = load_sample()

if raw_df is None:
    st.title("🏥 Health Analytics Dashboard")
    st.info("👈 Upload a CSV file or check 'Use sample dataset' in the sidebar to get started.")
    st.stop()

# --------------------------------------------------------------------------
# Validate & clean
# --------------------------------------------------------------------------
is_valid, missing_cols = validate_columns(raw_df)
if not is_valid:
    st.error(
        f"Uploaded file is missing required column(s): {', '.join(missing_cols)}.\n\n"
        "Please check the README for the expected data format."
    )
    st.stop()

df = clean_data(raw_df)
df = enrich_dataframe(df)

if df.empty:
    st.warning("No valid rows found after cleaning the data. Please check your file.")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar — Filters
# --------------------------------------------------------------------------
st.sidebar.markdown("### 🔎 Filters")

gender_opts = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", gender_opts)

disease_opts = ["All"] + sorted(df["Disease"].dropna().unique().tolist())
selected_disease = st.sidebar.selectbox("Disease", disease_opts)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
selected_age_range = st.sidebar.slider(
    "Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max)
)

filtered_df = df.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_disease != "All":
    filtered_df = filtered_df[filtered_df["Disease"] == selected_disease]
filtered_df = filtered_df[
    (filtered_df["Age"] >= selected_age_range[0])
    & (filtered_df["Age"] <= selected_age_range[1])
]

if filtered_df.empty:
    st.warning("No records match the selected filters. Try widening your filter selection.")
    st.stop()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏥 Health Analytics Dashboard")
st.caption(
    f"Showing **{len(filtered_df)}** of **{len(df)}** patient records "
    f"(filters applied: Gender = {selected_gender}, Disease = {selected_disease}, "
    f"Age = {selected_age_range[0]}–{selected_age_range[1]})"
)

# --------------------------------------------------------------------------
# KPI Cards
# --------------------------------------------------------------------------
st.markdown("### 📌 Key Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

total_patients = len(filtered_df)
avg_age = filtered_df["Age"].mean()
avg_bmi = filtered_df["BMI"].mean()
avg_systolic = filtered_df["BP_Systolic"].mean()
avg_diastolic = filtered_df["BP_Diastolic"].mean()

k1.metric("Total Patients", f"{total_patients}")
k2.metric("Average Age", f"{avg_age:.1f} yrs")
k3.metric("Average BMI", f"{avg_bmi:.1f}")
k4.metric("Avg Blood Pressure", f"{avg_systolic:.0f}/{avg_diastolic:.0f} mmHg")

if "Sentiment_Polarity" in filtered_df.columns:
    avg_sentiment = filtered_df["Sentiment_Polarity"].mean()
    sentiment_text = (
        "🙂 Positive" if avg_sentiment > 0.1
        else "🙁 Negative" if avg_sentiment < -0.1
        else "😐 Neutral"
    )
    k5.metric("Avg Feedback Sentiment", sentiment_text, f"{avg_sentiment:.2f}")
else:
    most_common_disease = filtered_df["Disease"].mode().iloc[0]
    k5.metric("Most Common Disease", most_common_disease)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Tabs for the different analysis sections
# --------------------------------------------------------------------------
tab_age, tab_disease, tab_bmi, tab_bp, tab_gender, tab_sentiment = st.tabs(
    ["👥 Age Distribution", "🦠 Disease Distribution", "⚖️ BMI Analysis",
     "❤️ Blood Pressure", "🚻 Gender Analysis", "💬 Feedback Sentiment"]
)

# ---------------------------- AGE DISTRIBUTION ---------------------------
with tab_age:
    st.subheader("Age Distribution")
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = px.histogram(
            filtered_df, x="Age", nbins=20,
            color_discrete_sequence=[PRIMARY_COLOR],
            title="Distribution of Patient Ages"
        )
        fig.update_layout(bargap=0.05, xaxis_title="Age", yaxis_title="Number of Patients")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        age_group_counts = filtered_df["Age_Group"].value_counts().reset_index()
        age_group_counts.columns = ["Age Group", "Count"]
        fig2 = px.pie(
            age_group_counts, names="Age Group", values="Count", hole=0.45,
            title="Patients by Age Group",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------- DISEASE DISTRIBUTION --------------------------
with tab_disease:
    st.subheader("Disease Distribution")
    disease_counts = filtered_df["Disease"].value_counts().reset_index()
    disease_counts.columns = ["Disease", "Count"]

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(
            disease_counts.sort_values("Count"), x="Count", y="Disease",
            orientation="h", color="Count", color_continuous_scale="Blues",
            title="Number of Patients per Disease"
        )
        fig.update_layout(yaxis_title="", xaxis_title="Number of Patients")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.pie(
            disease_counts, names="Disease", values="Count",
            title="Disease Share (%)",
            color_discrete_sequence=px.colors.sequential.Teal
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Disease Prevalence by Age Group")
    cross = pd.crosstab(filtered_df["Age_Group"], filtered_df["Disease"])
    fig3 = px.imshow(
        cross, text_auto=True, color_continuous_scale="Blues",
        aspect="auto", title="Heatmap: Age Group vs Disease"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------ BMI ANALYSIS ------------------------------
with tab_bmi:
    st.subheader("BMI Analysis")
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = px.histogram(
            filtered_df, x="BMI", nbins=25, color="BMI_Category",
            title="BMI Distribution by Category",
            color_discrete_map={
                "Underweight": "#60A5FA",
                "Normal": "#34D399",
                "Overweight": "#FBBF24",
                "Obese": "#F87171",
            }
        )
        fig.update_layout(bargap=0.05, xaxis_title="BMI", yaxis_title="Number of Patients")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bmi_cat_counts = filtered_df["BMI_Category"].value_counts().reset_index()
        bmi_cat_counts.columns = ["Category", "Count"]
        fig2 = px.pie(
            bmi_cat_counts, names="Category", values="Count", hole=0.45,
            title="BMI Category Share",
            color="Category",
            color_discrete_map={
                "Underweight": "#60A5FA",
                "Normal": "#34D399",
                "Overweight": "#FBBF24",
                "Obese": "#F87171",
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### BMI vs Age (bubble size = BMI)")
    fig3 = px.scatter(
        filtered_df, x="Age", y="BMI", color="BMI_Category", size="BMI",
        hover_data=["Name", "Gender", "Disease"],
        title="BMI vs Age Scatter Plot",
        color_discrete_map={
            "Underweight": "#60A5FA",
            "Normal": "#34D399",
            "Overweight": "#FBBF24",
            "Obese": "#F87171",
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------- BLOOD PRESSURE -------------------------------
with tab_bp:
    st.subheader("Blood Pressure Analysis")
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = px.scatter(
            filtered_df, x="BP_Systolic", y="BP_Diastolic", color="BP_Category",
            hover_data=["Name", "Age", "Gender"],
            title="Systolic vs Diastolic Blood Pressure",
            color_discrete_map={
                "Normal": "#34D399",
                "Elevated": "#FBBF24",
                "Hypertension Stage 1": "#FB923C",
                "Hypertension Stage 2": "#F87171",
            }
        )
        fig.add_vline(x=120, line_dash="dash", line_color="gray")
        fig.add_hline(y=80, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bp_counts = filtered_df["BP_Category"].value_counts().reset_index()
        bp_counts.columns = ["Category", "Count"]
        fig2 = px.pie(
            bp_counts, names="Category", values="Count", hole=0.45,
            title="Blood Pressure Category Share",
            color="Category",
            color_discrete_map={
                "Normal": "#34D399",
                "Elevated": "#FBBF24",
                "Hypertension Stage 1": "#FB923C",
                "Hypertension Stage 2": "#F87171",
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Average Blood Pressure by Age Group")
    bp_by_age = filtered_df.groupby("Age_Group")[["BP_Systolic", "BP_Diastolic"]].mean().reset_index()
    fig3 = go.Figure()
    fig3.add_bar(x=bp_by_age["Age_Group"], y=bp_by_age["BP_Systolic"], name="Systolic", marker_color=PRIMARY_COLOR)
    fig3.add_bar(x=bp_by_age["Age_Group"], y=bp_by_age["BP_Diastolic"], name="Diastolic", marker_color=ACCENT_COLOR)
    fig3.update_layout(barmode="group", title="Average Systolic/Diastolic BP by Age Group",
                        yaxis_title="mmHg")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------ GENDER ANALYSIS ---------------------------
with tab_gender:
    st.subheader("Gender Analysis")
    c1, c2 = st.columns([1, 2])

    with c1:
        gender_counts = filtered_df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.pie(
            gender_counts, names="Gender", values="Count", hole=0.4,
            title="Gender Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.box(
            filtered_df, x="Gender", y="BMI", color="Gender",
            title="BMI Distribution by Gender",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Disease Count by Gender")
    gender_disease = filtered_df.groupby(["Disease", "Gender"]).size().reset_index(name="Count")
    fig3 = px.bar(
        gender_disease, x="Disease", y="Count", color="Gender", barmode="group",
        title="Disease Distribution Split by Gender",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig3.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------- FEEDBACK SENTIMENT --------------------------
with tab_sentiment:
    st.subheader("Patient Feedback Sentiment (powered by TextBlob)")
    if "Sentiment_Label" not in filtered_df.columns:
        st.info("No 'Feedback' column found in the dataset — sentiment analysis is unavailable.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            sent_counts = filtered_df["Sentiment_Label"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(
                sent_counts, names="Sentiment", values="Count", hole=0.45,
                title="Overall Feedback Sentiment",
                color="Sentiment",
                color_discrete_map={"Positive": "#34D399", "Neutral": "#94A3B8", "Negative": "#F87171"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.histogram(
                filtered_df, x="Sentiment_Polarity", nbins=20,
                color="Sentiment_Label", title="Sentiment Polarity Distribution",
                color_discrete_map={"Positive": "#34D399", "Neutral": "#94A3B8", "Negative": "#F87171"}
            )
            fig2.update_layout(xaxis_title="Polarity Score (-1 = negative, +1 = positive)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Sample Feedback with Sentiment")
        st.dataframe(
            filtered_df[["Name", "Disease", "Feedback", "Sentiment_Label", "Sentiment_Polarity"]]
            .sort_values("Sentiment_Polarity")
            .reset_index(drop=True),
            use_container_width=True,
            height=300
        )

# --------------------------------------------------------------------------
# Raw data viewer + download
# --------------------------------------------------------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
with st.expander("📄 View Filtered Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data as CSV", data=csv,
        file_name="filtered_patient_data.csv", mime="text/csv"
    )

st.caption("Built with Streamlit, Plotly and TextBlob | Health Analytics Dashboard")
