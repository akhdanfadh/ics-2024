import sys
import time
import cv2
import mediapipe as mp
import numpy as np

# Adjustable parameters
DEFAULT_CAP_SRC = "data/person1.mp4"
PERCLOS_DURATION = 10
PERCLOS_THRESHOLD = 0.3
YAWN_THRESHOLD = 0.7
DROWSINESS_FRAMES = 30
CALIBRATION_FRAMES = 100
EAR_THRESHOLD_FACTOR = 0.8

# Landmark points
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 82, 311, 291, 402, 87]


def landmark_to_point(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def calculate_ear(eye_landmarks):
    points = [landmark_to_point(landmark) for landmark in eye_landmarks]
    vertical_dist1 = calculate_distance(points[1], points[5])
    vertical_dist2 = calculate_distance(points[2], points[4])
    horizontal_dist = calculate_distance(points[0], points[3])
    ear = (vertical_dist1 + vertical_dist2) / (2 * horizontal_dist + 1e-6)
    return ear


def detect_yawn(mouth_landmarks):
    points = [landmark_to_point(landmark) for landmark in mouth_landmarks]
    vertical_dist = calculate_distance(points[1], points[4])
    horizontal_dist = calculate_distance(points[0], points[3])
    mar = vertical_dist / (horizontal_dist + 1e-6)
    return mar > YAWN_THRESHOLD, mar


def calibrate_ear_threshold(cap, face_mesh):
    ear_values = []
    for _ in range(CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear([face_landmarks[i] for i in LEFT_EYE])
            right_ear = calculate_ear([face_landmarks[i] for i in RIGHT_EYE])
            avg_ear = (left_ear + right_ear) / 2
            ear_values.append(avg_ear)

    if ear_values:
        return np.mean(ear_values) * EAR_THRESHOLD_FACTOR
    else:
        return 0.25  # Fallback to default if calibration fails


def main(cap: cv2.VideoCapture):
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / original_fps if original_fps > 0 else 0.03
    perclos_window_size = int(PERCLOS_DURATION * original_fps)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    alert_counter = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("Calibrating EAR threshold...")
        ear_threshold = calibrate_ear_threshold(cap, face_mesh)
        print(f"Calibrated EAR threshold: {ear_threshold:.4f}")

        perclos_window = []
        drowsiness_counter = 0
        yawn_counter = 0
        start_time = time.time()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # EAR Analysis
                left_ear = calculate_ear([face_landmarks[i] for i in LEFT_EYE])
                right_ear = calculate_ear([face_landmarks[i] for i in RIGHT_EYE])
                avg_ear = (left_ear + right_ear) / 2

                # PERCLOS
                perclos_window.append(1 if avg_ear < ear_threshold else 0)
                if len(perclos_window) > perclos_window_size:
                    perclos_window.pop(0)
                perclos = sum(perclos_window) / len(perclos_window)

                # Yawning
                yawning, mar = detect_yawn([face_landmarks[i] for i in MOUTH])

                # Drowsiness detection logic
                if yawning:
                    drowsiness_counter += 1
                else:
                    drowsiness_counter = max(0, drowsiness_counter - 1)

                if yawning:
                    yawn_counter += 1
                else:
                    yawn_counter = max(0, yawn_counter - 1)

                # Drowsiness alert
                alert_reasons = []
                if drowsiness_counter > DROWSINESS_FRAMES:
                    alert_reasons.append(f"Yawning")
                if perclos > PERCLOS_THRESHOLD:
                    alert_reasons.append(f"PERCLOS ({perclos:.2f})")

                if alert_reasons:
                    alert_text = f"DROWSINESS ALERT! Reason: {' and '.join(alert_reasons)}"
                    cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_counter += 1

                # Display metrics
                cv2.putText(frame, f"EAR: {avg_ear:.2f}, MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawning: {'Yes' if yawning else 'No'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR Threshold: {ear_threshold:.4f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw facial landmarks
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

            cv2.imshow("Drowsiness Detection", frame)

            elapsed = time.time() - start_time
            time_to_wait = max(1, int((frame_count * frame_duration - elapsed) * 1000))

            if cv2.waitKey(time_to_wait) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total frames: {frame_count}")
    print(f"Total alerts: {alert_counter}")
    print(f"Percentage of time alert: {alert_counter / frame_count:.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap = cv2.VideoCapture(int(sys.argv[1]))
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        else:
            cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(DEFAULT_CAP_SRC)

    main(cap)
