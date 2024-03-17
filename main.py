import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
from pathlib import Path

# Constants
FRAME_DURATION = timedelta(milliseconds=33)  # Approximate duration of each frame in real-time capture

# Initialize variables for tracking
eye_closed_duration = timedelta(0)
eye_closed_start_time = None
drowsiness_detected_frames = 0

# Paths
base_dir = Path(__file__).resolve().parent
eye_model_path = base_dir / "pre-trained-models/eye_model.h5"
face_model_path = base_dir / "pre-trained-models/face_model.h5"

# Load pre-trained models
eye_model = load_model(str(eye_model_path))
face_model = load_model(str(face_model_path))

# Haar cascades for eye and face detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame, eye_cascade, face_cascade, eye_model, face_model):
    global eye_closed_duration, eye_closed_start_time, drowsiness_detected_frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eye_closed = False
    drowsy = False

    for (ex, ey, ew, eh) in eyes:
        eye = gray[ey:ey + eh, ex:ex + ew]
        eye = cv2.resize(eye, (24, 24))
        eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        eye = np.expand_dims(eye, axis=0) / 255.0
        prediction = eye_model.predict(eye)[0]
        if prediction < 0.5:  # The model outputs 0 for closed, 1 for open
            if eye_closed_start_time is None:
                eye_closed_start_time = datetime.now()
            eye_closed = True
            break

    if not eye_closed and eye_closed_start_time is not None:
        eye_closed_duration += datetime.now() - eye_closed_start_time
        eye_closed_start_time = None

    for (fx, fy, fw, fh) in faces:
        face = gray[fy:fy + fh, fx:fx + fw]
        face = cv2.resize(face, (150, 150))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = np.expand_dims(face, axis=0) / 255.0
        drowsy_pred = face_model.predict(face)[0]
        if drowsy_pred > 0.5:  # The model outputs >0.5 for drowsy
            drowsy = True
            drowsiness_detected_frames += 1

    return eye_closed, drowsy, eye_closed_duration

def test_from_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    image_paths = list(dataset_path.glob('*.jpg'))

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        eye_closed, drowsy, _ = process_frame(frame, eye_cascade, face_cascade, eye_model, face_model)
        print(f"Processed {image_path.name}: Eye Closed = {eye_closed}, Drowsy = {drowsy}")


def test_from_webcam():
    cap = cv2.VideoCapture(0)  # Open the default camera
    eye_closed_start_time = None
    drowsiness_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        eye_closed, drowsy, _ = process_frame(frame, eye_cascade, face_cascade, eye_model, face_model)
        
        # Check and update eye closed duration
        if eye_closed:
            if eye_closed_start_time is None:
                eye_closed_start_time = datetime.now()  # Eyes just closed
            eye_closed_duration = datetime.now() - eye_closed_start_time
            eye_closed_seconds = eye_closed_duration.total_seconds()
            cv2.putText(frame, f"Eyes closed for {eye_closed_seconds:.1f} seconds", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_closed_start_time = None

        # Check drowsiness and display drowsy alert if drowsy state persists
        if drowsy:
            if drowsiness_start_time is None:
                drowsiness_start_time = datetime.now()  # Drowsiness detected
            cv2.putText(frame, "Drowsiness detected!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            drowsiness_start_time = None

        # Display the resulting frame with the detection
        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    
def main():
    mode = input("Enter 'webcam' to test from webcam or 'dataset' to test from a dataset: ").strip().lower()

    if mode == 'webcam':
        print("Starting webcam mode. Press 'Q' to quit.")
        test_from_webcam()
    elif mode == 'dataset':
        dataset_dir = os.path.join(base_dir, "dataset")
        test_from_dataset(dataset_dir)
    else:
        print("Invalid mode. Please enter 'webcam' or 'dataset'.")

    # Display summary of the session
    print(f"Total eye closed duration (approx.): {eye_closed_duration}")
    print(f"Frames detected with drowsiness: {drowsiness_detected_frames}")

if __name__ == "__main__":
    main()
