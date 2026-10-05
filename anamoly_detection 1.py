import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1. LOAD & EXPLORE DATASET
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(script_dir, "*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in this folder. Place your G-Flix dataset here.")

csv_path = csv_files[0]
print(f"[*] Loading G-Flix activity logs from: {os.path.basename(csv_path)}")
df = pd.read_csv(csv_path)

print(f"[*] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n--- Data Structure ---")
print(df.info())

# Feature Engineering
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

# Encode categorical 'remote_access' (Yes=1, No=0)
if 'remote_access' in df.columns:
    df['remote_access_num'] = df['remote_access'].map({'Yes': 1, 'No': 0}).fillna(0)

# Behavioral feature set
behavioral_cols = ['login_duration_min', 'data_accessed_MB', 'files_downloaded', 'remote_access_num', 'hour']
behavioral_cols = [c for c in behavioral_cols if c in df.columns]

print(f"[*] Features used for anomaly profiling: {behavioral_cols}")
X = df[behavioral_cols].copy()

# =====================================================================
# 2. METHOD 1: STATISTICAL ANOMALY DETECTION (Z-SCORE)
# =====================================================================
print("\n" + "="*50)
print("[*] Running Method 1: Statistical Z-Score Detection")
print("="*50)

# Compute Z-Scores on numeric continuous metrics
stat_features = ['login_duration_min', 'data_accessed_MB', 'files_downloaded']
z_scores = np.abs((df[stat_features] - df[stat_features].mean()) / df[stat_features].std())

df['max_zscore'] = z_scores.max(axis=1)
df['stat_anomaly'] = df['max_zscore'] > 3.0

print(f"-> Records flagged by Statistical Z-Score (|Z| > 3): {df['stat_anomaly'].sum()}")

# =====================================================================
# 3. METHOD 2: UNSUPERVISED ML (ISOLATION FOREST)
# =====================================================================
print("\n" + "="*50)
print("[*] Running Method 2: Unsupervised ML (Isolation Forest)")
print("="*50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest with contamination matching expected outlier level (~1.5%)
iso_forest = IsolationForest(n_estimators=150, contamination=0.015, random_state=42)
iso_preds = iso_forest.fit_predict(X_scaled)

df['iso_anomaly'] = iso_preds == -1
df['anomaly_score'] = -iso_forest.decision_function(X_scaled)  # higher = more anomalous

print(f"-> Records flagged by Isolation Forest: {df['iso_anomaly'].sum()}")

# High-Confidence Suspects (Flagged by both approaches)
df['high_confidence'] = df['stat_anomaly'] & df['iso_anomaly']
print(f"-> High-Confidence Suspects (Flagged by Both Methods): {df['high_confidence'].sum()}")

# =====================================================================
# 4. RANK & DISPLAY TOP 5 SUSPECTS (ASCII SAFE)
# =====================================================================
suspects_df = df.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
top_5_suspects = suspects_df.head(5)

print("\n" + "="*65)
print("                    TOP 5 SUSPECTS DOSSIER")
print("="*65)

display_cols = ['user_id', 'timestamp', 'login_duration_min', 'data_accessed_MB', 'files_downloaded', 'remote_access', 'anomaly_score']
display_cols = [c for c in display_cols if c in top_5_suspects.columns]

print(top_5_suspects[display_cols].to_string(index=False))

# =====================================================================
# 5. VISUALIZATION
# =====================================================================
plt.figure(figsize=(13, 5))

# Plot 1: Behavioral Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(
    df.loc[~df['iso_anomaly'], 'data_accessed_MB'],
    df.loc[~df['iso_anomaly'], 'files_downloaded'],
    c='#2563eb', alpha=0.5, label='Normal Activity', s=35
)
plt.scatter(
    df.loc[df['iso_anomaly'], 'data_accessed_MB'],
    df.loc[df['iso_anomaly'], 'files_downloaded'],
    c='#dc2626', alpha=0.9, edgecolors='black', label='Flagged Anomalies', s=80
)

# Label top suspects
for i in range(min(5, len(top_5_suspects))):
    row = top_5_suspects.iloc[i]
    plt.annotate(
        f"#{i+1} ({row['user_id']})",
        (row['data_accessed_MB'], row['files_downloaded']),
        textcoords="offset points", xytext=(5, 5),
        fontsize=8, fontweight='bold', color='#991b1b'
    )

plt.xlabel('Data Accessed (MB)', fontsize=10)
plt.ylabel('Files Downloaded', fontsize=10)
plt.title('G-Flix Behavioral Profiling: Data vs Downloads', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Anomaly Score Distribution
plt.subplot(1, 2, 2)
plt.hist(df['anomaly_score'], bins=35, color='#475569', edgecolor='black', alpha=0.7)
threshold_score = top_5_suspects['anomaly_score'].iloc[-1]
plt.axvline(threshold_score, color='#dc2626', linestyle='--', linewidth=2, label='Top 5 Threshold')
plt.xlabel('Isolation Forest Anomaly Score', fontsize=10)
plt.ylabel('Activity Frequency', fontsize=10)
plt.title('Distribution of Behavioral Anomaly Severity', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
output_chart = os.path.join(script_dir, "anomaly_forensics_report.png")
plt.savefig(output_chart, dpi=300)
print(f"\n[*] Forensic plot saved as: {output_chart}")
plt.show()