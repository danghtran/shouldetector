import pandas as pd
import numpy as np

# Load your CSV (which has x_i, y_i, z_i for each landmark)
df = pd.read_csv("landmark_data/landmarks_fixation_temporal.csv")

def compute_3d_gaze(row, eye_corner, iris_points):
    # Eye corner points (3D)
    c1 = np.array([row[f'x{eye_corner[0]}'], row[f'y{eye_corner[0]}'], row[f'z{eye_corner[0]}']])
    c2 = np.array([row[f'x{eye_corner[1]}'], row[f'y{eye_corner[1]}'], row[f'z{eye_corner[1]}']])
    eye_center = (c1 + c2) / 2.0

    # Iris center (mean of iris points)
    iris = np.array([[row[f'x{i}'], row[f'y{i}'], row[f'z{i}']] for i in iris_points])
    iris_center = iris.mean(axis=0)

    # Gaze vector and normalization
    gaze_vec = iris_center - eye_center
    norm = np.linalg.norm(gaze_vec)
    if norm == 0:
        norm = 1e-6
    gaze_norm = gaze_vec / norm
    return gaze_norm

# Define landmark sets
left_eye = [33, 133]
right_eye = [362, 263]
left_iris = list(range(468, 473))
right_iris = list(range(473, 478))

# Compute for each row
gaze_data = df.apply(lambda r: np.concatenate([
    compute_3d_gaze(r, left_eye, left_iris),
    compute_3d_gaze(r, right_eye, right_iris)
]), axis=1, result_type='expand')

gaze_data.columns = ['gaze_x_left', 'gaze_y_left', 'gaze_z_left',
                     'gaze_x_right', 'gaze_y_right', 'gaze_z_right']

# Add to dataframe
df = pd.concat([df, gaze_data], axis=1)

# Save new CSV
df.to_csv("landmark_data/landmarks_fixation_temporal.csv", index=False)
