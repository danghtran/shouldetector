import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# ========== CONFIG ==========
TEST_CSV = "shoulder_data_test.csv"               # Your test dataset
MODEL_PATH = "xgb_model.pkl"            # XGBoost/RandomForest/etc.
SCALER_PATH = "scaler.pkl"          # Rebuilt scaler
LABEL_COL = "label"                 # Label column
TIMESTAMP_COL = "timestamp"         # Optional non-feature
FACEID_COL = "face_id"              # Optional non-feature
# ============================


print("✅ Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler_bundle = joblib.load(SCALER_PATH)
scaler = scaler_bundle["scaler"]
feature_cols = scaler_bundle["features"]

print("✅ Loading test dataset...")
df_test = pd.read_csv(TEST_CSV)

# Extract labels
y_test = df_test[LABEL_COL].map({"intentional": 1, "non-intentional": 0}).values

# Remove non-feature cols
drop_cols = [LABEL_COL]
for col in drop_cols:
    if col in df_test.columns:
        df_test = df_test.drop(columns=[col], errors='ignore')

# Ensure the feature column ordering is exactly the same as training
X_test = df_test[feature_cols].values

print("✅ Scaling test data...")
X_test_scaled = scaler.transform(X_test)

print("✅ Running predictions...")
pred_proba = model.predict_proba(X_test_scaled)[:, 1]
pred = (pred_proba >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, pred_proba)
cm = confusion_matrix(y_test, pred)

print("\n========== TEST REPORT ==========")
print(classification_report(y_test, pred, target_names=["non-intentional", "intentional"]))
print("Accuracy:", acc)
print("AUC:", auc)

print("\nConfusion Matrix:")
print(cm)

print("\n✅ Test completed.")
