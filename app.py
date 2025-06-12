import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# -------------------------------
# Load Precomputed Results
# -------------------------------
@st.cache_data
def load_results():
    df = pd.read_csv("results_summary.csv")
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    return df, metrics

df, metrics = load_results()

val_scores = df[df["label"] == 0]["score"]
attack_scores = df[df["label"] == 1]["score"]
threshold = np.percentile(val_scores, 95)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🔐 Intrusion Detection using AutoEncoder")
st.markdown("""
This app detects anomalies in network traffic using an autoencoder model trained on normal traffic patterns.  
High reconstruction error indicates potential cyber attacks.
""")

st.subheader("📁 Sample of Precomputed Traffic Summary")
st.dataframe(df.sample(10))

# -------------------------------
#  Anomaly Score Distribution
# -------------------------------
st.subheader("📉 Anomaly Score Distribution (Log Scale)")

fig, ax = plt.subplots()
ax.hist(val_scores, bins=100, alpha=0.7, label='Normal')
ax.hist(attack_scores, bins=100, alpha=0.7, label='Attack')
ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
ax.set_xscale("log")
ax.set_xlabel("Reconstruction Error (Log Scale)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Pie Chart / Prediction Summary
# -------------------------------
st.subheader("🧪 Prediction Summary")

y_pred = df["predicted"]
anomaly_counts = y_pred.value_counts().rename({0: "Normal", 1: "Anomaly"})
st.write(anomaly_counts)
st.bar_chart(anomaly_counts)

# -------------------------------
# Evaluation Metrics
# -------------------------------
st.subheader("📏 Evaluation Report")
st.write(f"**Precision (Anomaly):** {metrics['precision']:.2f}")
st.write(f"**Recall (Anomaly):** {metrics['recall']:.2f}")
st.write(f"**F1 Score (Anomaly):** {metrics['f1']:.2f}")
st.write(f"**AUC Score:** {metrics['auc']:.2f}")


# -------------------------------
st.subheader("📊 Anomaly Score Summary Stats (Filtered)")
val_filtered = val_scores[val_scores < np.percentile(val_scores, 99.9)]
attack_filtered = attack_scores[attack_scores < np.percentile(attack_scores, 99.9)]

col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Error (Normal)", f"{val_filtered.mean():.4f}")
    st.metric("Max Error (Normal)", f"{val_filtered.max():.2f}")
with col2:
    st.metric("Mean Error (Attack)", f"{attack_filtered.mean():.4f}")
    st.metric("Max Error (Attack)", f"{attack_filtered.max():.2f}")

st.caption("Note: Outliers above 99.9th percentile were excluded from the summary stats.")

# -------------------------------
# Show Top Anomalies
# -------------------------------
st.subheader("🚨 Top Detected Anomalies")

top_anomalies = df[df["predicted"] == 1].sort_values(by="score", ascending=False).head(10)
st.dataframe(top_anomalies)
