import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from collections import deque
from ultralytics import YOLO

# ---------- CONFIG ----------
CSV_FILE = "shoulder_data.csv"
FIXATION_WINDOW = 15          # frames for fixation computation
FIXATION_THRESHOLD = 3.0      # threshold (degrees)
YOLO_MODEL = "yolov8n.pt"

# ---------- INIT ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
model = YOLO(YOLO_MODEL)
cap = cv2.VideoCapture(0)

# current label mode
current_label = "safe"

# CSV header
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "face_id", "yaw", "pitch", "distance", "fixation_percent", "label"])

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

# ---------- CONSTANTS ----------
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
NOSE_TIP_IDX = 1

# track gaze history
face_histories = {}

print("Press 'S' for SAFE mode, 'A' for ATTACK mode, 'Q' to quit")

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
                    nose_tip = lm[NOSE_TIP_IDX]

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

                    with open(CSV_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.time(), face_id, yaw_avg, pitch_avg, distance_metric, fixation_percent, current_label
                        ])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"{current_label.upper()}",
                                (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                (0, 255, 0) if current_label=="safe" else (0, 0, 255), 2)
                    cv2.putText(frame, f"Fix:{fixation_percent:.0f}% Dist:{distance_metric:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    else:
        cv2.putText(frame, "Only one face detected (user)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Mode: {current_label.upper()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if current_label == "safe" else (0, 0, 255), 2)

    cv2.imshow("Shoulder Surfing Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        current_label = "safe"
    elif key == ord('a'):
        current_label = "attack"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
