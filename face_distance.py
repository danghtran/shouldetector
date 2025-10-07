import pandas as pd
import numpy as np

# Load your saved face mesh data
df = pd.read_csv("landmark_data/landmarks_fixation_temporal.csv")

# Key indices for reference distances
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
MOUTH_CENTER = 13

def get_point(df_row, idx):
    return np.array([df_row[f"x{idx}"], df_row[f"y{idx}"], df_row[f"z{idx}"]])

def compute_face_distance(df_row):
    # (1) Distance between eyes
    left_eye = get_point(df_row, LEFT_EYE_OUTER)
    right_eye = get_point(df_row, RIGHT_EYE_OUTER)
    eye_distance = np.linalg.norm(left_eye - right_eye)

    # (2) Nose to mouth distance
    nose_tip = get_point(df_row, NOSE_TIP)
    mouth_center = get_point(df_row, MOUTH_CENTER)
    nose_mouth_dist = np.linalg.norm(nose_tip - mouth_center)

    # (3) Mean Z depth across key features
    mean_z = np.mean([left_eye[2], right_eye[2], nose_tip[2], mouth_center[2]])

    # Combine into a relative distance metric
    # Smaller eye_distance ⇒ face is farther
    # More negative mean_z ⇒ face is closer
    distance_metric = (1 / (eye_distance + 1e-6)) + mean_z

    return round(distance_metric, 6)

# Apply to all rows
df["face_distance_metric"] = df.apply(compute_face_distance, axis=1)

# Save back
df.to_csv("landmark_data/landmarks_fixation_temporal.csv", index=False)
