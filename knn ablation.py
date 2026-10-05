import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================================
# 1. LOAD & PREPROCESS DATA
# =====================================================================
# Ensure path dynamically finds data.csv in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "data.csv")

if not os.path.exists(csv_file):
    # Fallback to current working directory
    csv_file = "data.csv"

print(f"Loading dataset from: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Original shape: {df.shape}")

# Requirement 1: Drop 'id'
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Remove empty/unnamed trailing column if present from Kaggle export
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

# Requirement 2: Encode diagnosis (M = 1, B = 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df = df.dropna().reset_index(drop=True)

# Separate features (X) and label (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].values
feature_names = list(X.columns)

print(f"Cleaned dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class counts: Malignant (1) = {sum(y == 1)}, Benign (0) = {sum(y == 0)}")

# Train/Test Split (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# 2. BASELINE KNN MODEL (ALL FEATURES, NORMALIZED)
# =====================================================================
K_VALUE = 5

# Requirement 3: Normalize features before KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline
baseline_knn = KNeighborsClassifier(n_neighbors=K_VALUE)
baseline_knn.fit(X_train_scaled, y_train)
y_pred_base = baseline_knn.predict(X_test_scaled)

base_acc = accuracy_score(y_test, y_pred_base)
base_prec = precision_score(y_test, y_pred_base)
base_rec = recall_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base)

print("\n================ BASELINE KNN MODEL (k=5, ALL FEATURES) ================")
print(f"Accuracy : {base_acc:.4f}")
print(f"Precision: {base_prec:.4f}")
print(f"Recall   : {base_rec:.4f}")
print(f"F1-Score : {base_f1:.4f}")

# =====================================================================
# 3. FEATURE ABLATION STUDY
# =====================================================================
print("\nRunning feature ablation study (removing one feature at a time)...")
ablation_records = []

for feat in feature_names:
    # 1. Remove one feature at a time
    X_train_sub = X_train.drop(columns=[feat])
    X_test_sub = X_test.drop(columns=[feat])

    # 2. Re-scale the ablated feature subset
    sc = StandardScaler()
    X_train_sub_sc = sc.fit_transform(X_train_sub)
    X_test_sub_sc = sc.transform(X_test_sub)

    # 3. Retrain with consistent k=5
    knn = KNeighborsClassifier(n_neighbors=K_VALUE)
    knn.fit(X_train_sub_sc, y_train)
    y_pred = knn.predict(X_test_sub_sc)

    # 4. Compare all 4 metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    ablation_records.append({
        'Removed_Feature': feat,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Delta_Accuracy': acc - base_acc,
        'Delta_F1': f1 - base_f1
    })

ablation_df = pd.DataFrame(ablation_records)

# =====================================================================
# 4. ANALYZE IMPACT
# =====================================================================
# Sort by largest drop in F1-score (most negative delta = most critical feature)
ranked_features = ablation_df.sort_values(by='Delta_F1', ascending=True)

print("\n========== TOP 10 MOST CRITICAL FEATURES (LARGEST DROP WHEN REMOVED) ==========")
print(ranked_features[['Removed_Feature', 'Accuracy', 'F1_Score', 'Delta_Accuracy', 'Delta_F1']].head(10).to_string(index=False))

# =====================================================================
# 5. VISUALIZATION
# =====================================================================
# Visualize top 10 features whose removal drops performance most
top_dropped = ranked_features.head(10).copy()

plt.figure(figsize=(11, 6))
bars = plt.barh(top_dropped['Removed_Feature'], top_dropped['Delta_F1'] * 100, color='#d95f02')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
plt.gca().invert_yaxis()
plt.xlabel('Change in F1-Score (%) Relative to Baseline', fontsize=11)
plt.title(f'KNN Feature Ablation: Impact of Removing Individual Features (k={K_VALUE})', fontsize=13)
plt.grid(axis='x', linestyle=':', alpha=0.6)

# Annotate values on the bars
for bar in bars:
    w = bar.get_width()
    plt.text(w - 0.08 if w < 0 else w + 0.05, bar.get_y() + bar.get_height()/2,
             f"{w:.2f}%", va='center', ha='right' if w < 0 else 'left', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('knn_ablation_results.png', dpi=300)
plt.show()