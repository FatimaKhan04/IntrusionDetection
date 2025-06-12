import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# -------------------------------
# 🎯 Page Setup
# -------------------------------
st.set_page_config(page_title="Intrusion Detection", layout="centered")
st.title("🔐 Intrusion Detection using AutoEncoder (Precomputed)")
st.markdown("""
This app demonstrates intrusion detection using an autoencoder model.
All results shown here are based on precomputed anomaly scores and visualizations.
""")

# -------------------------------
# 📊 Load Precomputed Data
# -------------------------------
@st.cache_data
def load_results():
    df = pd.read_csv("results_summary.csv")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return df, metrics

df, metrics = load_results()

# -------------------------------
# 📉 Anomaly Score Distribution (Precomputed)
# -------------------------------
st.subheader("📉 Anomaly Score Distribution")

st.image("anomaly_distribution.png", caption="Histogram (Log Scale) of reconstruction errors with threshold")

# -------------------------------
# 📊 Prediction Summary
# -------------------------------
st.subheader("🧪 Prediction Summary")
label_counts = df["predicted"].value_counts().rename({0: "Normal", 1: "Anomaly"})
st.write(label_counts)
st.bar_chart(label_counts)

# -------------------------------
# 📏 Evaluation Metrics
# -------------------------------
st.subheader("📏 Evaluation Report")
col1, col2 = st.columns(2)
with col1:
    st.metric("Precision (Anomaly)", f"{metrics['precision']:.2f}")
    st.metric("F1 Score (Anomaly)", f"{metrics['f1']:.2f}")
with col2:
    st.metric("Recall (Anomaly)", f"{metrics['recall']:.2f}")
    st.metric("AUC Score", f"{metrics['auc']:.2f}")

# -------------------------------
# 🔍 Show Top Anomalies (Optional)
# -------------------------------
st.subheader("🚨 Top Detected Anomalies (High Scores)")
top_anomalies = df[df["predicted"] == 1].sort_values(by="score", ascending=False).head(10)
st.dataframe(top_anomalies)
