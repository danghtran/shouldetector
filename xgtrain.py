import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib

# ============================================================
# 1. Load Data
# ============================================================
TRAIN_PATH = "train_aug.csv"
TEST_PATH = "shoulder_data.csv"
SCALER_OUT = "scaler.pkl"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"Loaded {len(df_train)} training rows, {len(df_test)} test rows.")

# Label → binary
df_train["label"] = df_train["label"].map({"intentional": 1, "non-intentional": 0})
df_test["label"] = df_test["label"].map({"intentional": 1, "non-intentional": 0})

y_train = df_train["label"].values
y_test = df_test["label"].values

X_train = df_train.drop(columns=["label"])
X_test = df_test.drop(columns=["label"])

# ============================================================
# 2. Scaling
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# drop_cols = ["label"]
# feature_cols = [c for c in df_train.columns if c not in drop_cols]
# joblib.dump({"scaler": scaler, "features": feature_cols}, SCALER_OUT)

# ============================================================
# 3. Build XGBoost model (NO EARLY STOP)
# ============================================================
def build_model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.10,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1
    )

# ============================================================
# 4. 5-Fold Stratified CV
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
auc_scores = []

print("\n========== XGBoost 5-Fold Cross Validation ==========\n")

for fold, (tr, val) in enumerate(skf.split(X_train_scaled, y_train), start=1):
    print(f"========== Fold {fold}/5 ==========")

    X_tr, X_val = X_train_scaled[tr], X_train_scaled[val]
    y_tr, y_val = y_train[tr], y_train[val]

    model = build_model()

    # ✅ NO early stopping, just basic training
    model.fit(X_tr, y_tr, verbose=False)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)

    acc_scores.append(acc)
    auc_scores.append(auc)

    print(f"Accuracy: {acc:.3f} | AUC: {auc:.3f}")
    print(classification_report(
        y_val,
        preds,
        target_names=["non-intentional", "intentional"]
    ))

# ============================================================
# 5. Summary
# ============================================================
print("\n========== Cross-Validation Summary ==========")
print(f"Mean Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"Mean AUC:      {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

# ============================================================
# 6. Final model (NO early stopping)
# ============================================================
final_model = build_model()
final_model.fit(X_train_scaled, y_train, verbose=False)

# ============================================================
# 7. Evaluation on Test Set
# ============================================================
test_preds = final_model.predict(X_test_scaled)
test_probs = final_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)

print("\n========== FINAL TEST PERFORMANCE ==========")
print(classification_report(
    y_test,
    test_preds,
    target_names=["non-intentional", "intentional"]
))
print(f"Accuracy: {test_acc:.3f}")
print(f"AUC: {test_auc:.3f}")

joblib.dump(final_model, "xgb_model.pkl")
