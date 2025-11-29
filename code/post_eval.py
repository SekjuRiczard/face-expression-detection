"""Post-training evaluation: confusion matrix visualization and
One-vs-Rest ROC curve plotting.

Loads test dataset, model and label mapping. Generates:
    - Confusion Matrix (matplotlib)
    - ROC curves for each class
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.utils import to_categorical

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(BASE, "artifacts", "mini-xception")
TEST = os.path.join(BASE, "images", "test")

LABELS_PATH = os.path.join(BASE, "artifacts", "labels.json")
MODEL_PATH  = os.path.join(BASE, "artifacts", "mini-xception", "best.keras")

IMG_SIZE = (64, 64)


def load_images_test(root):
    """Load grayscale test images from directory and normalize them.

        Args:
            root (str): Path to dataset folder.

        Returns:
            tuple[np.ndarray, list[str]]:
                - Array of images shaped (N, 64, 64, 1)
                - List of string class labels
        """
    paths, labels = [], []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            paths.append(os.path.join(d, f))
            labels.append(label)

    X, y = [], []
    for p in paths:
        img = tf.keras.preprocessing.image.load_img(
            p, color_mode="grayscale", target_size=IMG_SIZE
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        X.append(arr[..., None])
    return np.array(X), labels



if __name__ == "__main__":

    # labels.json
    with open(LABELS_PATH, "r") as f:
        class_names = json.load(f)

    # load test set
    X_test, y_labels = load_images_test(TEST)
    y_true = np.array([class_names.index(x) for x in y_labels])
    y_true_cat = to_categorical(y_true, num_classes=len(class_names))

    # load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # predictions
    probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_cat[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.grid(True)
    plt.show()
