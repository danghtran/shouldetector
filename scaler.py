import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ========== CONFIG ==========
TRAIN_CSV = "train_aug.csv"       # your full training dataset
SCALER_OUT = "scaler.pkl"
LABEL_COL = "label"
TIMESTAMP_COL = "timestamp"
FACEID_COL = "face_id"
# ============================

print("Loading data...")
df = pd.read_csv(TRAIN_CSV)

# Drop non-feature columns
drop_cols = [LABEL_COL]
feature_cols = [c for c in df.columns if c not in drop_cols]

print(f"Detected {len(feature_cols)} feature columns.")
print("Features:", feature_cols)

X = df[feature_cols].astype(float)

print("Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

joblib.dump({"scaler": scaler, "features": feature_cols}, SCALER_OUT)

print("✅ Scaler rebuilt and saved to:", SCALER_OUT)
print("✅ Now your real-time pipeline will match training normalization.")
