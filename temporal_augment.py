"""
v4_feature_pipeline_aggregated.py
----------------------------------
Fixed version:
 - NO sliding window
 - NO time-warping of frames
 - NO per-frame augmentation
 - Works directly on aggregated window rows from your capture script
 - Augmentation operates ONLY on aggregated features

Usage:

# Augmented training
python v4_feature_pipeline_aggregated.py --input shoulder_data.csv --mode train_augmented --out augmented_v4.csv

# Original deterministic (clean) windows
python v4_feature_pipeline_aggregated.py --input shoulder_data.csv --mode train_original --out original_clean.csv

# Eval/test windows (no augmentation)
python v4_feature_pipeline_aggregated.py --input test_data.csv --mode eval --out eval_processed.csv
"""

import argparse
import numpy as np
import pandas as pd
import random

LABEL_COL = "label"

# All aggregated feature columns (MUST match your format)
FEATURE_COLS = [
    # coords
    "left_eye_x_mean","left_eye_x_std",
    "left_eye_y_mean","left_eye_y_std",
    "left_eye_z_mean","left_eye_z_std",
    "right_eye_x_mean","right_eye_x_std",
    "right_eye_y_mean","right_eye_y_std",
    "right_eye_z_mean","right_eye_z_std",
    "nose_x_mean","nose_x_std",
    "nose_y_mean","nose_y_std",
    "nose_z_mean","nose_z_std",

    # yaw/pitch/distance/fixation means/stds
    "yaw_mean_mean","yaw_mean_std",
    "pitch_mean_mean","pitch_mean_std",
    "distance_mean_mean","distance_mean_std",
    "fixation_mean_mean","fixation_mean_std",

    # per-frame std aggregated columns
    "yaw_std_mean","yaw_std_std",
    "pitch_std_mean","pitch_std_std",
    "distance_std_mean","distance_std_std",
    "fixation_std_mean","fixation_std_std",

    # dynamics
    "vel_mean_mean","vel_mean_std",
    "vel_std_mean","vel_std_std",
    "acc_mean_mean","acc_mean_std",
    "acc_std_mean","acc_std_std",

    # smoothness
    "smoothness_mean","smoothness_std"
]

# ---------------------------
# Label-conditional augmentation for aggregated features
# ---------------------------

def augment_intent(row):
    r = row.copy()
    # intentional: more stable, smoother, fewer saccades
    r["yaw_mean_std"] *= np.random.uniform(0.6, 0.9)
    r["pitch_mean_std"] *= np.random.uniform(0.6, 0.9)
    r["vel_mean_mean"] *= np.random.uniform(0.9, 1.05)
    r["acc_mean_mean"] *= np.random.uniform(0.9, 1.05)
    r["smoothness_mean"] *= np.random.uniform(1.1, 1.4)
    return r

def augment_nonintent(row):
    r = row.copy()
    # non-intentional: noisier, more saccades
    r["yaw_mean_std"] *= np.random.uniform(1.2, 1.8)
    r["pitch_mean_std"] *= np.random.uniform(1.2, 1.8)
    r["vel_mean_mean"] *= np.random.uniform(1.1, 1.4)
    r["acc_mean_mean"] *= np.random.uniform(1.1, 1.4)
    r["smoothness_mean"] *= np.random.uniform(0.6, 0.85)
    return r

def deterministic_clean(row):
    # fully deterministic: no randomness
    return row.copy()

# ---------------------------
# MAIN PROCESS
# ---------------------------

def process_file(input_csv, mode="eval", out_csv=None, aug_factor=4):
    df = pd.read_csv(input_csv)

    new_rows = []

    for _, row in df.iterrows():
        # always include original–clean for training
        base = deterministic_clean(row)

        if mode == "train_original":
            new_rows.append(base)

        elif mode == "train_augmented":

            # original deterministic
            new_rows.append(base)

            # augmented copies
            for _ in range(aug_factor):
                if row[LABEL_COL] == "intentional":
                    new_rows.append(augment_intent(base))
                else:
                    new_rows.append(augment_nonintent(base))

        elif mode == "eval":
            new_rows.append(base)

        else:
            raise ValueError("Unknown mode: " + mode)

    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {len(out_df)} samples → {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["train_augmented", "train_original", "eval"], default="eval")
    p.add_argument("--aug_factor", type=int, default=4)
    args = p.parse_args()

    # deterministic seed for eval
    if args.mode in ["eval", "train_original"]:
        random.seed(42)
        np.random.seed(42)

    process_file(args.input, mode=args.mode, out_csv=args.out, aug_factor=args.aug_factor)


if __name__ == "__main__":
    main()
