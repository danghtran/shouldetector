import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from collections import deque
from ultralytics import YOLO

# ---------- CONFIG ----------
CSV_FILE = "shoulder_data.csv"
WINDOW_SIZE = 15
STEP_SIZE = 10
YOLO_MODEL = "yolov8n.pt"
FIXATION_WINDOW = 5
FIXATION_THRESHOLD = 3.0

# ---------- INIT ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
model = YOLO(YOLO_MODEL)
cap = cv2.VideoCapture(0)

current_mode = "safe"  # "safe" → intentional, "attack" → non-intentional
frame_buffer = deque(maxlen=WINDOW_SIZE)
frame_count = 0
face_histories = {}

# ---------- CSV HEADER ----------
with open(CSV_FILE, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "face_id",
        # landmark raw positions (averaged per window)
        "left_eye_x", "left_eye_y", "left_eye_z",
        "right_eye_x", "right_eye_y", "right_eye_z",
        "nose_x", "nose_y", "nose_z",
        # motion and fixation stats
        "yaw_mean", "pitch_mean", "distance_mean", "fixation_mean",
        "yaw_std", "pitch_std", "distance_std", "fixation_std",
        "vel_mean", "vel_std", "acc_mean", "acc_std", "smoothness",
        "label"
    ])

# ---------- HELPERS ----------
def compute_gaze_angles(eye_center, nose_tip):
    vec = eye_center - nose_tip
    vec /= np.linalg.norm(vec)
    yaw = np.degrees(np.arctan2(vec[0], -vec[2]))
    pitch = np.degrees(np.arctan2(vec[1], -vec[2]))
    return yaw, pitch

def eye_center(landmarks, indices):
    pts = np.array([landmarks[i] for i in indices])
    return np.mean(pts, axis=0)

def compute_fixation_percentage(history):
    if len(history) < 2:
        return 0.0
    yaws, pitches = zip(*history)
    yaw_std = np.std(yaws)
    pitch_std = np.std(pitches)
    stable = (yaw_std < FIXATION_THRESHOLD) and (pitch_std < FIXATION_THRESHOLD)
    fixation_percent = np.clip((1 - (yaw_std + pitch_std) / (2 * FIXATION_THRESHOLD)) * 100, 0, 100)
    return fixation_percent if stable else fixation_percent / 2

def compute_temporal_dynamics(yaw_pitch_series):
    yaws, pitches = np.array(yaw_pitch_series).T
    vel = np.sqrt(np.diff(yaws)**2 + np.diff(pitches)**2)
    acc = np.diff(vel)
    vel_mean, vel_std = np.mean(vel), np.std(vel)
    acc_mean, acc_std = np.mean(acc), np.std(acc)
    smoothness = 1 / (vel_std + 1e-6)
    return vel_mean, vel_std, acc_mean, acc_std, smoothness

LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
NOSE_TIP_IDX = 1

# ---------- MAIN LOOP ----------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    faces = []
    for det in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, det)
        area = (x2 - x1) * (y2 - y1)
        faces.append((area, (x1, y1, x2, y2)))

    faces.sort(key=lambda f: f[0], reverse=True)

    if len(faces) > 1:
        # Ignore largest (user) face
        for face_id, (_, (x1, y1, x2, y2)) in enumerate(faces[1:], start=1):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(crop_rgb)

            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    h, w, _ = crop.shape
                    lm = [(p.x * w, p.y * h, p.z) for p in face_landmarks.landmark]

                    left_eye = eye_center(lm, LEFT_EYE_IDX)
                    right_eye = eye_center(lm, RIGHT_EYE_IDX)
                    nose_tip = np.array(lm[NOSE_TIP_IDX])

                    yaw_l, pitch_l = compute_gaze_angles(left_eye, nose_tip)
                    yaw_r, pitch_r = compute_gaze_angles(right_eye, nose_tip)
                    yaw_avg = (yaw_l + yaw_r) / 2
                    pitch_avg = (pitch_l + pitch_r) / 2

                    left_outer = np.array(lm[33])
                    right_outer = np.array(lm[263])
                    eye_dist = np.linalg.norm(left_outer - right_outer)
                    distance_metric = 1 / (eye_dist + 1e-6)

                    if face_id not in face_histories:
                        face_histories[face_id] = deque(maxlen=FIXATION_WINDOW)
                    face_histories[face_id].append((yaw_avg, pitch_avg))
                    fixation_percent = compute_fixation_percentage(face_histories[face_id])

                    # Add frame data to window
                    frame_buffer.append([
                        yaw_avg, pitch_avg, distance_metric, fixation_percent,
                        *left_eye, *right_eye, *nose_tip
                    ])
                    frame_count += 1

                    if len(frame_buffer) == WINDOW_SIZE and frame_count % STEP_SIZE == 0:
                        arr = np.array(frame_buffer)
                        mean_vals = np.mean(arr[:, :4], axis=0)
                        std_vals = np.std(arr[:, :4], axis=0)
                        vel_mean, vel_std, acc_mean, acc_std, smoothness = compute_temporal_dynamics(arr[:, :2])

                        # Average 3D points
                        left_mean = np.mean(arr[:, 4:7], axis=0)
                        right_mean = np.mean(arr[:, 7:10], axis=0)
                        nose_mean = np.mean(arr[:, 10:13], axis=0)

                        label = "intentional" if current_mode == "safe" else "non-intentional"

                        with open(CSV_FILE, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.time(), face_id,
                                *left_mean, *right_mean, *nose_mean,
                                *mean_vals, *std_vals,
                                vel_mean, vel_std, acc_mean, acc_std, smoothness,
                                label
                            ])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"{current_mode.upper()}",
                                (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if current_mode == "safe" else (0, 0, 255), 2)
                    cv2.putText(frame, f"Fix:{fixation_percent:.0f}% Dist:{distance_metric:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, f"Mode: {current_mode.upper()} ({'Intentional' if current_mode=='safe' else 'Non-intentional'})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if current_mode == "safe" else (0, 0, 255), 2)

    cv2.imshow("Intent Window Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        current_mode = "safe"
    elif key == ord('a'):
        current_mode = "attack"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
