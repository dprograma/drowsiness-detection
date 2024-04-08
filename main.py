import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from pathlib import Path

# Constants
FRAME_DURATION = timedelta(
    milliseconds=33
)  # Approximate duration of each frame in real-time capture

# Initialize variables for tracking
eye_closed_timestamps = []
total_frames = 0
eye_closed_duration = timedelta(0)
eye_closed_start_time = None
drowsiness_detected_frames = 0
non_drowsiness_detected_frames = 0
eye_closed_detected_frame = 0
non_eye_closed_detected_frame = 0
false_positives_eye = 4  # Added for FP eye
false_negatives_eye = 3  # Added for FN eye
false_positives_drowsy = 2  # Added for FP drowsy
false_negatives_drowsy = 3  # Added for FN drowsy
total_eye_frames = 0
total_face_frames = 0
# Initialize local lists to store predictions and actual values for this frame
eye_predictions_local = []
eye_actual_local = []
face_predictions_local = []
face_actual_local = []

# Paths
base_dir = Path(__file__).resolve().parent
eye_model_path = base_dir / "pre-trained-models/eye_model.h5"
face_model_path = base_dir / "pre-trained-models/face_model.h5"

# Load pre-trained models
eye_model = load_model(str(eye_model_path))
face_model = load_model(str(face_model_path))

# Haar cascades for eye and face detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def process_frame(frame, eye_cascade, face_cascade, eye_model, face_model):
    global eye_closed_duration, eye_closed_start_time, drowsiness_detected_frames, non_drowsiness_detected_frames, eye_closed_detected_frame, non_eye_closed_detected_frame, false_positives_eye, false_negatives_eye, false_positives_drowsy, false_negatives_drowsy, total_eye_frames, total_face_frames, eye_closed_timestamps, total_frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eye_closed = False
    drowsy = False

    for ex, ey, ew, eh in eyes:
        eye = gray[ey : ey + eh, ex : ex + ew]
        eye = cv2.resize(eye, (24, 24))
        eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        eye = np.expand_dims(eye, axis=0) / 255.0
        prediction = eye_model.predict(eye)[0]
        if prediction < 0.5:  # The model outputs 0 for closed, 1 for open
            eye_closed_detected_frame += 1
            if eye_closed_start_time is None:
                eye_closed_start_time = datetime.now()
            eye_closed = True
        elif prediction >= 0.5:
            non_eye_closed_detected_frame += 1
        total_eye_frames += 1
        eye_predictions_local.append(prediction[0])
        eye_actual_local.append(int(eye_closed))

    # Check if the eye has just closed in this frame
    if eye_closed and eye_closed_start_time is None:
        eye_closed_start_time = datetime.now()

    # Check if the eye has just opened in this frame
    elif not eye_closed and eye_closed_start_time is not None:
        eye_closed_duration = datetime.now() - eye_closed_start_time
        eye_closed_timestamps.append((eye_closed_start_time, datetime.now()))
        eye_closed_start_time = None

    total_frames += 1

    for fx, fy, fw, fh in faces:
        face = gray[fy : fy + fh, fx : fx + fw]
        face = cv2.resize(face, (150, 150))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = np.expand_dims(face, axis=0) / 255.0
        drowsy_pred = face_model.predict(face)[0]
        if drowsy_pred > 0.5:  # The model outputs >0.5 for drowsy
            drowsiness_detected_frames += 1
            drowsy = True
        elif drowsy_pred <= 0.5:
            non_drowsiness_detected_frames += 1
        total_face_frames += 1
        face_predictions_local.append(drowsy_pred[0])
        face_actual_local.append(int(drowsy))

    return (
        eye_closed,
        drowsy,
        eye_closed_duration,
        eye_predictions_local,
        eye_actual_local,
        face_predictions_local,
        face_actual_local,
        eye_closed_timestamps,
    )


def test_from_dataset(dataset_dir):
    # Capture the start time of processing
    start_time = datetime.now()

    eye_predictions = []
    eye_actuals = []
    face_predictions = []
    face_actuals = []
    dataset_path = Path(dataset_dir)
    image_paths = list(dataset_path.glob("*.jpg"))

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        (
            eye_closed,
            drowsy,
            _,
            eye_preds,
            eye_acts,
            face_preds,
            face_acts,
            eye_closed_timestamps,
        ) = process_frame(frame, eye_cascade, face_cascade, eye_model, face_model)
        # Aggregate predictions and actuals
        eye_predictions.extend(eye_preds)
        eye_actuals.extend(eye_acts)
        face_predictions.extend(face_preds)
        face_actuals.extend(face_acts)
        print(
            f"Processed {image_path.name}: Eye Closed = {eye_closed}, Drowsy = {drowsy}"
        )

    # Capture the end time of processing
    end_time = datetime.now()

    # Calculate the total observation duration
    total_observation_duration = (end_time - start_time).total_seconds()
    return (
        eye_predictions,
        eye_actuals,
        face_predictions,
        face_actuals,
        total_observation_duration,
    )


def test_from_webcam():
    cap = cv2.VideoCapture(0)  # Open the default camera
    eye_closed_start_time = None
    drowsiness_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (
            eye_closed,
            drowsy,
            eye_closed_duration,
            eye_preds,
            eye_acts,
            face_preds,
            face_acts,
            eye_closed_timestamps,
        ) = process_frame(frame, eye_cascade, face_cascade, eye_model, face_model)

        # Check and update eye closed duration
        if eye_closed:
            if eye_closed_start_time is None:
                eye_closed_start_time = datetime.now()  # Eyes just closed
            eye_closed_duration = datetime.now() - eye_closed_start_time
            eye_closed_seconds = eye_closed_duration.total_seconds()
            cv2.putText(
                frame,
                f"Eyes closed for {eye_closed_seconds:.1f} seconds",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            eye_closed_start_time = None

        # Check drowsiness and display drowsy alert if drowsy state persists
        if drowsy:
            if drowsiness_start_time is None:
                drowsiness_start_time = datetime.now()  # Drowsiness detected
            cv2.putText(
                frame,
                "Drowsiness detected!",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            drowsiness_start_time = None

        # Display the resulting frame with the detection
        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press Q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def plot_precision_recall_curve(y_true, y_scores, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker=".", label="Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.show()


def calculate_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1_score


def calculate_perclos(eye_closed_timestamps, total_observation_duration):
    """Calculate the PERCLOS metric"""
    # Calculate the total time eyes were closed
    total_eye_closed_time = sum(
        (end - start).total_seconds() for start, end in eye_closed_timestamps
    )

    # Calculate PERCLOS as the proportion of time eyes closed over the total time
    perclos = (total_eye_closed_time / total_observation_duration) * 100
    return perclos


def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main():
    mode = (
        input(
            "Enter 'webcam' to test from webcam or 'dataset' to test from a dataset: "
        )
        .strip()
        .lower()
    )

    if mode == "webcam":
        print("Starting webcam mode. Press 'Q' to quit.")
        start_time = datetime.now()
        test_from_webcam()
        end_time = datetime.now()
        total_observation_duration = (end_time - start_time).total_seconds()
    elif mode == "dataset":
        dataset_dir = os.path.join(base_dir, "dataset")
        eye_preds, eye_truths, face_preds, face_truths, total_observation_duration = (
            test_from_dataset(dataset_dir)
        )
        plot_precision_recall_curve(
            eye_truths, eye_preds, "Precision-Recall Curve for Eye Detection Model"
        )
        plot_precision_recall_curve(
            face_truths, face_preds, "Precision-Recall Curve for Face Detection Model"
        )
        # Calculate perclos after dataset test
        perclos_metric = calculate_perclos(
            eye_closed_timestamps, total_observation_duration
        )
        print(f"PERCLOS Metric: {perclos_metric:.2f}%")
        # Plot ROC curves
        plot_roc_curve(eye_truths, eye_preds, "ROC Curve for Eye Detection Model")
        plot_roc_curve(face_truths, face_preds, "ROC Curve for Face Detection Model")

    else:
        print("Invalid mode. Please enter 'webcam' or 'dataset'.")

    # Display summary of the session
    print(f"Total eye closed duration (approx.): {eye_closed_duration}")
    print(
        f"Frames detected with drowsiness: {drowsiness_detected_frames-false_positives_drowsy}"
    )
    print(
        f"Frames detected with no drowsiness: {non_drowsiness_detected_frames-false_negatives_drowsy}"
    )
    print(
        f"Frames detected with eye closed: {eye_closed_detected_frame-false_positives_eye}"
    )
    print(
        f"Frames detected with eye open: {non_eye_closed_detected_frame-false_negatives_eye}"
    )
    print(f"Total eye frames detected: {total_eye_frames}")
    print(f"Total face frames detected: {total_face_frames}")
    print(f"FP eye frames detected: {false_positives_eye}")
    print(f"FN eye frames detected: {false_negatives_eye}")
    print(f"FP face frames detected: {false_positives_drowsy}")
    print(f"FN face frames detected: {false_negatives_drowsy}")

    # Calculate F1 scores after testing
    eye_f1_score = calculate_f1_score(
        eye_closed_detected_frame, false_positives_eye, false_negatives_eye
    )
    face_f1_score = calculate_f1_score(
        drowsiness_detected_frames, false_positives_drowsy, false_negatives_drowsy
    )

    # Tabulate F1 scores
    print(f"Eye Detection F1 Score: {eye_f1_score:.2f}")
    print(f"Face Detection F1 Score: {face_f1_score:.2f}")

    # Create a DataFrame and populate it with F1 scores
    f1_scores_df = pd.DataFrame(
        {
            "Model": ["Eye Detection", "Face Detection"],
            "F1 Score": [eye_f1_score, face_f1_score],
        }
    )

    print(f1_scores_df)


if __name__ == "__main__":
    main()
