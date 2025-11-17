import pandas as pd

TRAIN_CSV = "shoulder_temporal_augmented_p.csv"
TEST_CSV = "test_p.csv"

# Load datasets
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("==========================================")
print("🔍 LABEL DISTRIBUTION (TRAIN)")
print("==========================================")
print(train["label"].value_counts())
print("\nProportion:")
print(train["label"].value_counts(normalize=True))

print("\n==========================================")
print("🔍 LABEL DISTRIBUTION (TEST)")
print("==========================================")
print(test["label"].value_counts())
print("\nProportion:")
print(test["label"].value_counts(normalize=True))

# ---------------------------------------------------------
# Compare feature distributions (mean/std per column)
# ---------------------------------------------------------

# Drop label column
train_features = train.drop(columns=["label"])
test_features = test.drop(columns=["label"])

print("\n==========================================")
print("📊 FEATURE SUMMARY DIFFERENCES (TRAIN vs TEST)")
print("==========================================")

summary = pd.DataFrame({
    "train_mean": train_features.mean(),
    "test_mean": test_features.mean(),
    "train_std": train_features.std(),
    "test_std": test_features.std()
})

# Difference column
summary["mean_diff"] = (summary["test_mean"] - summary["train_mean"]).abs()
summary["std_ratio"] = summary["test_std"] / summary["train_std"]

print(summary)

# ---------------------------------------------------------
# Identify features with large shifts
# ---------------------------------------------------------
print("\n==========================================")
print("⚠️ FEATURES WITH LARGE MEAN SHIFT (>20% of train std)")
print("==========================================")

mean_shift_threshold = (summary["train_std"] * 0.2).fillna(0)
shifted = summary[summary["mean_diff"] > mean_shift_threshold]

print(shifted)

print("\n==========================================")
print("⚠️ FEATURES WITH LARGE STD RATIO (> 1.5x or < 0.66x)")
print("==========================================")

std_shifted = summary[(summary["std_ratio"] > 1.5) | (summary["std_ratio"] < 0.66)]
print(std_shifted)
