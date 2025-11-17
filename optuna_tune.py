import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ==========================================================
# Load Data
# ==========================================================
TRAIN_PATH = "shoulder_temporal_augmented_p.csv"

df = pd.read_csv(TRAIN_PATH)
df["label"] = df["label"].map({"intentional": 1, "non-intentional": 0})

y = df["label"].values
X = df.drop(columns=["label"]).values

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================================================
# Objective Function for Optuna
# ==========================================================
def objective(trial):
    # ✅ Search Space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []

    # ✅ 5-fold CV inside Optuna
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)

# ==========================================================
# Run Optuna Study
# ==========================================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n===== BEST PARAMETERS =====")
print(study.best_params)
print("Best AUC:", study.best_value)

# ==========================================================
# Train final model with best hyperparameters
# ==========================================================
best_params = study.best_params
best_params.update({
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1
})

final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# Save model
import joblib
joblib.dump(final_model, "xgb_optuna_best_model.pkl")

print("\n✅ Saved best XGBoost model as xgb_optuna_best_model.pkl")
