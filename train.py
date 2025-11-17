"""
training_v5.py

Train RandomForest & XGBoost on aggregated window-level data.

Input:
    train_ready_windows.csv

Output:
    model_rf.pkl
    model_xgb.pkl
    training_report.txt
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
INPUT_CSV = "shoulder_temporal_augmented_p.csv"
RF_MODEL_OUT = "model_rf.pkl"
XGB_MODEL_OUT = "model_xgb.pkl"
REPORT_OUT = "training_report.txt"

N_SPLITS = 5
RANDOM_STATE = 42


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded rows:", len(df))

# Label encode
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Features
X = df.drop(["label", "label_encoded"], axis=1)
y = df["label_encoded"]

feature_names = list(X.columns)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -------------------------------------------------------------
# STRATIFIED K-FOLD CROSS VALIDATION
# -------------------------------------------------------------
def evaluate_model(model, model_name):
    print(f"\n========== {model_name} Cross-Validation ==========")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_acc, fold_auc = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, prob)

        fold_acc.append(acc)
        fold_auc.append(auc)

        print(f"Fold {fold}/{N_SPLITS} | ACC={acc:.3f} | AUC={auc:.3f}")

    print("\nMean ACC:", np.mean(fold_acc), "±", np.std(fold_acc))
    print("Mean AUC:", np.mean(fold_auc), "±", np.std(fold_auc))

    return model


# -------------------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

rf = evaluate_model(rf, "RandomForest")

# FINAL TEST PERFORMANCE
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\n========== RANDOM FOREST TEST SET ==========")
print(classification_report(y_test, rf_preds, target_names=label_encoder.classes_))
print("ACC =", accuracy_score(y_test, rf_preds))
print("AUC =", roc_auc_score(y_test, rf_probs))


# -------------------------------------------------------------
# XGBOOST
# -------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)

xgb = evaluate_model(xgb, "XGBoost")

# TEST PERFORMANCE
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]

print("\n========== XGBOOST TEST SET ==========")
print(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_))
print("ACC =", accuracy_score(y_test, xgb_preds))
print("AUC =", roc_auc_score(y_test, xgb_probs))


# -------------------------------------------------------------
# CONFUSION MATRIX (RANDOM FOREST)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix – RandomForest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")


# -------------------------------------------------------------
# FEATURE IMPORTANCE (RANDOM FOREST)
# -------------------------------------------------------------
importances = rf.feature_importances_
idx = np.argsort(importances)[-20:]  # top 20 features

plt.figure(figsize=(7, 6))
plt.barh(np.array(feature_names)[idx], importances[idx])
plt.title("Top 20 Feature Importances (RandomForest)")
plt.tight_layout()
plt.savefig("rf_feature_importance_top20.png")


# -------------------------------------------------------------
# SAVE MODELS
# -------------------------------------------------------------
joblib.dump(rf, RF_MODEL_OUT)
joblib.dump(xgb, XGB_MODEL_OUT)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nSaved models:")
print(" -", RF_MODEL_OUT)
print(" -", XGB_MODEL_OUT)
print(" - scaler.pkl")
print(" - label_encoder.pkl")

print("\nTraining complete.")
