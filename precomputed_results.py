import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import json

# -------------------------------
# Load Data and Model
# -------------------------------
df = pd.read_csv('./data/NF-UNSW-NB15-v2.csv')

drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack']
features = [
    'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT'
]

df_cleaned = df.drop(columns=drop_cols)
labels = df_cleaned['Label']
df_cleaned = df_cleaned.drop(columns=['Label'])
df_cleaned = df_cleaned[features]

df_normal = df_cleaned[labels == 0]
df_attack = df_cleaned[labels == 1]

df_train, df_val = train_test_split(df_normal, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train)
X_val = scaler.transform(df_val)
X_attack = scaler.transform(df_attack)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_attack_tensor = torch.tensor(X_attack, dtype=torch.float32)

# -------------------------------
# Define Model
# -------------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(31, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 31)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# -------------------------------
# Load Trained Model
# -------------------------------
model = AutoEncoder()
model.load_state_dict(torch.load("autoencoder_model.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------------------
# Compute Anomaly Scores
# -------------------------------
def compute_scores(tensor_data):
    with torch.no_grad():
        recon = model(tensor_data)
        return ((tensor_data - recon) ** 2).mean(dim=1).numpy()

val_scores = compute_scores(X_val_tensor)
attack_scores = compute_scores(X_attack_tensor)

# Combine
all_scores = np.concatenate([val_scores, attack_scores])
y_true = np.array([0]*len(val_scores) + [1]*len(attack_scores))

# -------------------------------
# Compute Threshold & Predictions
# -------------------------------
threshold = np.percentile(val_scores, 95)
y_pred = all_scores > threshold


df_results = pd.DataFrame({
    "score": all_scores,
    "label": y_true,
    "predicted": y_pred.astype(int)
})
df_results.to_csv("results_summary.csv", index=False)
print("Saved: results_summary.csv")


report = classification_report(y_true, y_pred, target_names=["Normal", "Attack"], output_dict=True)
metrics = {
    "precision": report['Attack']['precision'],
    "recall": report['Attack']['recall'],
    "f1": report['Attack']['f1-score'],
    "auc": roc_auc_score(y_true, all_scores)
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(" Saved: metrics.json")

fig, ax = plt.subplots()
ax.hist(val_scores, bins=100, alpha=0.7, label='Normal')
ax.hist(attack_scores, bins=100, alpha=0.7, label='Attack')
ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
ax.set_xscale("log")
ax.set_xlabel("Reconstruction Error (Log Scale)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
fig.savefig("anomaly_distribution.png")
print(" Saved: anomaly_distribution.png")
