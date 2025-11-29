"""Realtime facial-expression recognition using webcam feed.

This module:
- Loads trained mini-Xception model.
- Captures webcam frames via OpenCV.
- Detects faces (HaarCascade).
- Preprocesses detected regions.
- Runs model prediction and overlays labels + confidence.

Requires:
    - OpenCV (cv2)
    - TensorFlow/Keras
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import json

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_PATH = os.path.join(BASE, "artifacts", "labels.json")
MODEL_PATH = os.path.join(BASE, "artifacts", "mini-xception", "best.keras")

with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)  # angry, fear, happy, neutral, sad, surprise

IMG_SIZE = (64, 64)


def preprocess(gray_face):
    """Resize, normalize and batch a face image for inference.

        Args:
            gray_face (np.ndarray): Grayscale face crop.

        Returns:
            np.ndarray: Tensor of shape (1, 64, 64, 1) normalized to [0,1].
        """
    resized = cv2.resize(gray_face, IMG_SIZE)
    arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # shape (1,64,64,1)
    return arr

if __name__ == "__main__":

    model = tf.keras.models.load_model(MODEL_PATH)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (DirectShow)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            x_input = preprocess(face_gray)
            probs = model.predict(x_input, verbose=0)[0]
            cls_id = int(np.argmax(probs))
            cls = CLASS_NAMES[cls_id]
            conf = float(np.max(probs))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{cls} ({conf:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        cv2.imshow("FER Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
