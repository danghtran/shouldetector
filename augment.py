import pandas as pd
import numpy as np
import random

INPUT_CSV = "shoulder_data.csv"
OUTPUT_CSV = "train_aug.csv"

# Controlled randomness
np.random.seed(42)
random.seed(42)

df = pd.read_csv(INPUT_CSV)

# List of coordinate columns (x, y, z)
coord_cols = [
    "left_eye_x","left_eye_y","left_eye_z",
    "right_eye_x","right_eye_y","right_eye_z",
    "nose_x","nose_y","nose_z"
]

# Other numeric features
signal_cols = [
    "yaw_mean","pitch_mean","distance_mean","fixation_mean",
    "yaw_std","pitch_std","distance_std","fixation_std",
    "vel_mean","vel_std","acc_mean","acc_std","smoothness"
]

def augment_row(row):
    aug = row.copy()

    # === 1) DOMAIN RANDOMIZATION ON COORDINATES ============================
    scale = np.random.uniform(0.75, 1.35)          # simulate sitting closer/further
    shift = np.random.uniform(-20, 20)             # simulate camera/laptop position change

    for col in coord_cols:
        aug[col] = row[col] * scale + shift

    # === 2) RANDOM CAMERA ANGLE VARIATION ==================================
    aug["yaw_mean"]  = row["yaw_mean"]  + np.random.normal(0, 5)
    aug["pitch_mean"] = row["pitch_mean"] + np.random.normal(0, 0.5)

    # === 3) DISTANCE AUGMENTATION ==========================================
    # Real data had distance_mean from 0.4 → 90+
    aug["distance_mean"] *= np.random.uniform(0.7, 1.5)
    aug["distance_mean"] += np.random.uniform(-4, 4)

    aug["distance_std"] *= np.random.uniform(0.7, 1.6)

    # === 4) FIXATION BEHAVIOR AUGMENTATION =================================
    aug["fixation_mean"] *= np.random.uniform(0.5, 2.0)
    aug["fixation_std"]  *= np.random.uniform(0.5, 2.5)

    # === 5) SACCADE + BURST SIMULATION =====================================
    # Shoulder surfing often causes rapid gaze changes
    if random.random() < 0.3:  # 30% probability add saccade burst
        burst_vel = np.random.uniform(8, 40)
        burst_acc = np.random.uniform(10, 60)
        aug["vel_mean"] += burst_vel
        aug["acc_mean"] += burst_acc

    # small baseline noise to all velocity/acceleration stats
    aug["vel_mean"] += np.random.normal(0, 2)
    aug["vel_std"]  += np.random.normal(0, 0.5)
    aug["acc_mean"] += np.random.normal(0, 2)
    aug["acc_std"]  += np.random.normal(0, 0.7)

    # === 6) SMOOTHNESS NOISE ===============================================
    aug["smoothness"] *= np.random.uniform(0.8, 1.3)

    # === 7) TIMESTAMP SHIFT (optional) =====================================
    aug["timestamp"] = row["timestamp"] + np.random.uniform(-0.03, 0.03)

    # Keep face_id & label the same
    return aug


# === AUGMENTATION MULTIPLIER ==============================================
AUG_TIMES = 3   # 1 original row → 3 synthetic rows

augmented_rows = []

for idx, row in df.iterrows():
    augmented_rows.append(row)  # original row
    for _ in range(AUG_TIMES):
        augmented_rows.append(augment_row(row))

aug_df = pd.DataFrame(augmented_rows)
aug_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Augmentation complete!")
print(f"Original rows: {len(df)}")
print(f"Augmented rows: {len(aug_df)} saved to {OUTPUT_CSV}")
