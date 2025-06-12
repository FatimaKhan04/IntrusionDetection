# 🔐 IntrusionDetection

This Streamlit app demonstrates an **anomaly-based intrusion detection system** using a deep learning **AutoEncoder** model trained on the [NF-UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) dataset.

The app analyzes network traffic and highlights potential intrusions based on reconstruction errors from a trained AutoEncoder. It displays precomputed metrics and visualizations for a clean, fast, and ready-to-deploy experience.

---

## 📂 What's Included

| File | Description |
|------|-------------|
| `app.py` | Streamlit app for visualizing results |
| `results_summary.csv` | Precomputed anomaly scores and predictions |
| `metrics.json` | Precision, recall, F1-score, and AUC |
| `anomaly_distribution.png` | Log-scaled histogram of anomaly scores |
| `.gitignore` | Excludes large datasets/models from Git |
| `requirements.txt` | Python dependencies |

---

## 🚀 Features

- 📉 Log-scale histogram of anomaly scores
- 📊 Precision, recall, F1, and AUC metrics
- 🧠 Highlights top 10 anomalies detected
- ✅ Fast deployment without loading large datasets
- ⚡️ Uses precomputed results — no model loading at runtime

---

## 🧪 How It Works

1. An AutoEncoder model was trained on **normal traffic only**.
2. It was then evaluated on both normal and attack traffic.
3. Reconstruction errors were used to classify anomalies.
4. All scores, predictions, and metrics were saved for this app.

---
