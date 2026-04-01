import time
import numpy as np
import cv2
import face_recognition
import mediapipe as mp

def benchmark():
    # Create a dummy frame (black image with a white square as "face")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (400, 300), (255, 255, 255), -1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. MediaPipe Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    start = time.time()
    results = face_detection.process(rgb_frame)
    mp_duration = time.time() - start
    print(f"MediaPipe Detection: {mp_duration:.4f}s")

    # 2. face_recognition.face_locations (HOG)
    start = time.time()
    face_locations = face_recognition.face_locations(rgb_frame)
    fr_duration = time.time() - start
    print(f"face_recognition.face_locations (HOG): {fr_duration:.4f}s")

    if results.detections:
        # Convert MediaPipe to face_recognition format
        h, w, _ = frame.shape
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        top = int(bbox.ymin * h)
        right = int((bbox.xmin + bbox.width) * w)
        bottom = int((bbox.ymin + bbox.height) * h)
        left = int(bbox.xmin * w)

        # 3. face_recognition.face_encodings with known locations
        start = time.time()
        encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
        encoding_duration = time.time() - start
        print(f"face_recognition.face_encodings (with locations): {encoding_duration:.4f}s")

    print(f"\nPotential speedup for detection: {fr_duration / mp_duration:.2f}x")

if __name__ == "__main__":
    benchmark()
