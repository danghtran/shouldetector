import pandas as pd
import numpy as np

# === 1. Load your saved landmark data ===
df = pd.read_csv("landmark_data/landmarks_fixation_temporal.csv")

# === 2. Define MediaPipe landmark indices for key points ===
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
NOSE_TIP_IDX = 1

# === 3. Helper functions ===
def get_point(df_row, idx):
    return np.array([
        df_row[f"x{idx}"],
        df_row[f"y{idx}"],
        df_row[f"z{idx}"]
    ])

def eye_center(df_row, indices):
    pts = np.array([get_point(df_row, i) for i in indices])
    return np.mean(pts, axis=0)

def compute_gaze_angles(eye_center, nose_tip):
    vec = eye_center - nose_tip
    norm = np.linalg.norm(vec)
    if norm == 0:
        return 0.0, 0.0
    vec /= norm
    yaw = np.degrees(np.arctan2(vec[0], -vec[2]))   # left-right
    pitch = np.degrees(np.arctan2(vec[1], -vec[2])) # up-down
    return yaw, pitch

# === 4. Process each row (frame) ===
yaw_list, pitch_list = [], []

for _, row in df.iterrows():
    try:
        left_eye = eye_center(row, LEFT_EYE_IDX)
        right_eye = eye_center(row, RIGHT_EYE_IDX)
        nose_tip = get_point(row, NOSE_TIP_IDX)

        yaw_l, pitch_l = compute_gaze_angles(left_eye, nose_tip)
        yaw_r, pitch_r = compute_gaze_angles(right_eye, nose_tip)

        yaw_avg = (yaw_l + yaw_r) / 2
        pitch_avg = (pitch_l + pitch_r) / 2

        yaw_list.append(yaw_avg)
        pitch_list.append(pitch_avg)
    except KeyError:
        yaw_list.append(np.nan)
        pitch_list.append(np.nan)

# === 5. Add to the same DataFrame and save ===
df["gaze_yaw"] = yaw_list
df["gaze_pitch"] = pitch_list

df.to_csv("landmark_data/landmarks_fixation_temporal.csv", index=False)
