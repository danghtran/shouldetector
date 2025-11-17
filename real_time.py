# real_time_classifier.py
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import csv
from collections import deque
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
CSV_LOG = "realtime_predictions.csv"   # optional log of each window prediction
YOLO_MODEL = "yolov8n.pt"
MODEL_FILE = "xgb_model.pkl"
SCALER_FILE = "scaler.pkl"             # can be either a scaler object or a dict {"scaler":..., "features": [...]}
WINDOW_SIZE = 15
STEP_SIZE = 10
FIXATION_WINDOW = 5
FIXATION_THRESHOLD = 3.0
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
NOSE_TIP_IDX = 1
# ---------------------------------------------

# ---------------- load model + scaler robustly ----------------
def load_model(path):
    with open(path, "rb") as f:
        return joblib.load(path)

def load_scaler_bundle(path):
    # Try many formats: dict with "scaler" and "features", or a scaler object itself
    try:
        b = joblib.load(path)
    except Exception:
            raise RuntimeError("Cannot load scaler file: " + path)

    if isinstance(b, dict) and "scaler" in b and "features" in b:
        return b["scaler"], b["features"]
    # if it's an sklearn scaler object, we don't have feature names — caller must rely on ordering
    return b, None

model = load_model(MODEL_FILE)
scaler, scaler_feature_order = load_scaler_bundle(SCALER_FILE)

# ---------------- expected feature order (must match training) ----------------
FEATURE_ORDER = [
    "timestamp","face_id",
    "left_eye_x","left_eye_y","left_eye_z",
    "right_eye_x","right_eye_y","right_eye_z",
    "nose_x","nose_y","nose_z",
    "yaw_mean","pitch_mean","distance_mean","fixation_mean",
    "yaw_std","pitch_std","distance_std","fixation_std",
    "vel_mean","vel_std","acc_mean","acc_std","smoothness"
]

# If scaler provided feature list, prefer it (keeps column ordering safe)
if scaler_feature_order is not None:
    FEATURE_ORDER = scaler_feature_order

# -------------- helpers (same as your capture) --------------
def compute_gaze_angles(eye_center, nose_tip):
    vec = eye_center - nose_tip
    n = np.linalg.norm(vec)
    if n == 0:
        return 0.0, 0.0
    vec = vec / n
    yaw = np.degrees(np.arctan2(vec[0], -vec[2]))
    pitch = np.degrees(np.arctan2(vec[1], -vec[2]))
    return float(yaw), float(pitch)

def eye_center(landmarks, indices):
    pts = np.array([landmarks[i] for i in indices], dtype=float)
    return np.mean(pts, axis=0)

def compute_fixation_percentage(history):
    if len(history) < 2:
        return 0.0
    yaws, pitches = zip(*history)
    yaw_std = np.std(yaws)
    pitch_std = np.std(pitches)
    stable = (yaw_std < FIXATION_THRESHOLD) and (pitch_std < FIXATION_THRESHOLD)
    fixation_percent = np.clip((1 - (yaw_std + pitch_std) / (2 * FIXATION_THRESHOLD)) * 100, 0, 100)
    return float(fixation_percent if stable else fixation_percent / 2)

def compute_temporal_dynamics(yaw_pitch_series):
    yaws, pitches = np.array(yaw_pitch_series).T
    if len(yaws) < 2:
        return 0.0, 0.0, 0.0, 0.0, 1.0
    vel = np.sqrt(np.diff(yaws)**2 + np.diff(pitches)**2)
    if len(vel) < 1:
        vel = np.array([0.0])
    acc = np.diff(vel) if len(vel) > 1 else np.array([0.0])
    vel_mean, vel_std = float(np.mean(vel)), float(np.std(vel))
    acc_mean, acc_std = float(np.mean(acc)), float(np.std(acc))
    smoothness = float(1.0 / (vel_std + 1e-6))
    return vel_mean, vel_std, acc_mean, acc_std, smoothness

# ---------------- init detectors ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
yolo = YOLO(YOLO_MODEL)
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=WINDOW_SIZE)
face_histories = {}
frame_count = 0

# Prepare CSV logging (header)
with open(CSV_LOG, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ["timestamp","face_id"] + FEATURE_ORDER + ["pred","prob","mode"]
    )

print("Real-time classifier started. Press 'q' to quit. 's' safe, 'a' attack (label toggle).")

# ---------------- main loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, verbose=False)
    faces = []
    for det in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, det)
        area = (x2 - x1) * (y2 - y1)
        faces.append((area, (x1, y1, x2, y2)))
    faces.sort(key=lambda f: f[0], reverse=True)

    if len(faces) > 1:
        # iterate attacker faces (skip largest user face)
        for face_id, (_, (x1, y1, x2, y2)) in enumerate(faces[1:], start=1):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(crop_rgb)

            if mesh_results.multi_face_landmarks:
                # Only use first detected mesh (should be single)
                lm = [(p.x * crop.shape[1], p.y * crop.shape[0], p.z) for p in mesh_results.multi_face_landmarks[0].landmark]

                left_eye = eye_center(lm, LEFT_EYE_IDX)
                right_eye = eye_center(lm, RIGHT_EYE_IDX)
                nose_tip = np.array(lm[NOSE_TIP_IDX], dtype=float)

                yaw_l, pitch_l = compute_gaze_angles(left_eye, nose_tip)
                yaw_r, pitch_r = compute_gaze_angles(right_eye, nose_tip)
                yaw_avg = (yaw_l + yaw_r) / 2.0
                pitch_avg = (pitch_l + pitch_r) / 2.0

                left_outer = np.array(lm[33], dtype=float)
                right_outer = np.array(lm[263], dtype=float)
                eye_dist = float(np.linalg.norm(left_outer - right_outer))
                distance_metric = float(1.0 / (eye_dist + 1e-6))

                # fixation history
                if face_id not in face_histories:
                    face_histories[face_id] = deque(maxlen=FIXATION_WINDOW)
                face_histories[face_id].append((yaw_avg, pitch_avg))
                fixation_percent = compute_fixation_percentage(face_histories[face_id])

                # append frame-level info (array layout matches your capture)
                frame_buffer.append([
                    yaw_avg, pitch_avg, distance_metric, fixation_percent,
                    float(left_eye[0]), float(left_eye[1]), float(left_eye[2]),
                    float(right_eye[0]), float(right_eye[1]), float(right_eye[2]),
                    float(nose_tip[0]), float(nose_tip[1]), float(nose_tip[2])
                ])
                frame_count += 1

                # when we have a full window and at STEP boundaries -> produce one aggregated sample (same as capture)
                if len(frame_buffer) == WINDOW_SIZE and frame_count % STEP_SIZE == 0:
                    arr = np.array(frame_buffer)  # shape (WINDOW, 13)
                    mean_vals = np.mean(arr[:, :4], axis=0)   # yaw,pitch,distance,fixation
                    std_vals = np.std(arr[:, :4], axis=0)
                    vel_mean, vel_std, acc_mean, acc_std, smoothness = compute_temporal_dynamics(arr[:, :2])

                    left_mean = np.mean(arr[:, 4:7], axis=0)
                    right_mean = np.mean(arr[:, 7:10], axis=0)
                    nose_mean = np.mean(arr[:, 10:13], axis=0)

                    # build feature dict in the FEATURE_ORDER sequence
                    row = {
                        "timestamp": time.time(), "face_id": face_id,
                        "left_eye_x": left_mean[0], "left_eye_y": left_mean[1], "left_eye_z": left_mean[2],
                        "right_eye_x": right_mean[0], "right_eye_y": right_mean[1], "right_eye_z": right_mean[2],
                        "nose_x": nose_mean[0], "nose_y": nose_mean[1], "nose_z": nose_mean[2],
                        "yaw_mean": mean_vals[0], "pitch_mean": mean_vals[1],
                        "distance_mean": mean_vals[2], "fixation_mean": mean_vals[3],
                        "yaw_std": std_vals[0], "pitch_std": std_vals[1],
                        "distance_std": std_vals[2], "fixation_std": std_vals[3],
                        "vel_mean": vel_mean, "vel_std": vel_std,
                        "acc_mean": acc_mean, "acc_std": acc_std,
                        "smoothness": smoothness
                    }

                    X_list = [row[f] for f in FEATURE_ORDER]
                    X = np.array([X_list], dtype=float)

                    # scale with robustness
                    try:
                        X_scaled = scaler.transform(X)
                    except Exception:
                        # if scaler was saved as dict with "scaler", handle earlier; fallback: use X raw
                        X_scaled = X

                    # predict
                    try:
                        proba = model.predict_proba(X_scaled)[0]
                        # if binary: proba[1] is intentional-class prob; otherwise choose index of 'intentional' if present
                        if hasattr(model, "classes_"):
                            classes = list(model.classes_)
                            if "intentional" in classes:
                                idx = classes.index("intentional")
                                prob_intent = float(proba[idx])
                            else:
                                # default to second column if binary
                                prob_intent = float(proba[-1])
                        else:
                            prob_intent = float(proba[-1])
                    except Exception:
                        # model may be non-proba; use decision_function if available
                        try:
                            val = model.decision_function(X_scaled)[0]
                            prob_intent = 1.0 / (1.0 + np.exp(-val))  # sigmoid
                        except Exception:
                            prob_intent = 0.0

                    # label determination (try numeric or string forms)
                    try:
                        pred_raw = model.predict(X_scaled)[0]
                        if isinstance(pred_raw, (int, np.integer)):
                            # numeric label: 1 -> intentional
                            pred_label = "intentional" if int(pred_raw) == 1 else "non-intentional"
                        else:
                            pred_label = str(pred_raw)
                    except Exception:
                        pred_label = "intentional" if prob_intent > 0.5 else "non-intentional"

                    # overlay and optional CSV log
                    cv2.putText(frame, f"{pred_label.upper()} ({prob_intent*100:.1f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0,255,0) if pred_label=="intentional" else (0,0,255), 2)

                    # write to CSV log
                    with open(CSV_LOG, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([time.time(), face_id] + X_list + [pred_label, prob_intent, "live"])

                    # move window forward by STEP_SIZE frames to mimic your capture saving behavior
                    # pop STEP_SIZE frames from left (this keeps buffer aligned to STEP)
                    for _ in range(STEP_SIZE):
                        if frame_buffer:
                            frame_buffer.popleft()

                # draw face box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                cv2.putText(frame, f"Fix:{fixation_percent:.0f}% Dist:{distance_metric:.2f}",
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # top-left status
    cv2.putText(frame, f"Press s=SAFE a=ATTACK q=QUIT", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.imshow("Realtime Shoulder Surfing Classifier", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # optional: you can toggle mode if you want to collect labeled data during demo
        pass
    elif key == ord('a'):
        pass

cap.release()
cv2.destroyAllWindows()
