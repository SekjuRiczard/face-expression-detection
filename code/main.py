"""Main entry point for full FER training pipeline.

Steps performed:
    1. Load image paths into DataFrame.
    2. Remove unwanted classes (e.g., 'disgust').
    3. Encode labels.
    4. Load image pixel data.
    5. Build mini-Xception model.
    6. Train with augmentation + class weights.
    7. Evaluate on test set.
    8. Save model artifacts (model + labels.json).
"""
import json
import os
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf

from model_mini_xception import build_mini_xception
from trainer import make_datasets, train

tf.keras.mixed_precision.set_global_policy("mixed_float16")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN = os.path.join(BASE, "images", "train")
TEST = os.path.join(BASE, "images", "test")
ART = os.path.join(BASE, "artifacts", "mini-xception")

IMG_SIZE = (64, 64)
BATCH = 64
EPOCHS = 150


def df_from_dir(directory: str) -> pd.DataFrame:
    """Create a DataFrame of file paths and labels from directory tree.

        Args:
            directory (str): Root folder containing label subfolders.

        Returns:
            pandas.DataFrame: Columns {path, label}.
        """
    imgs, labels = [], []
    for label in sorted(os.listdir(directory)):
        dir2 = os.path.join(directory, label)
        if not os.path.isdir(dir2):
            continue
        for fname in os.listdir(dir2):
            imgs.append(os.path.join(dir2, fname))
            labels.append(label)
    return pd.DataFrame({"path": imgs, "label": labels})


def load_images(paths: List[str], size=IMG_SIZE):
    """Load and normalize images from disk, converting to grayscale.

       Args:
           paths (list[str]): List of file paths.
           size (tuple[int, int]): Target resolution.

       Returns:
           np.ndarray: Image batch of shape (N, H, W, 1).
       """
    arr = []
    for p in tqdm(paths, desc="Loading images"):
        img = load_img(p, color_mode="grayscale", target_size=size)
        a = np.array(img, dtype=np.float32) / 255.0
        arr.append(a[..., None])
    return np.stack(arr)


if __name__ == "__main__":

    df_train = df_from_dir(TRAIN)
    df_test = df_from_dir(TEST)

    df_train = df_train[df_train.label != "disgust"]
    df_test = df_test[df_test.label != "disgust"]

    labels = sorted(df_train.label.unique())
    label_to_idx = {c: i for i, c in enumerate(labels)}

    y_train_idx = np.array([label_to_idx[x] for x in df_train.label])
    y_test_idx = np.array([label_to_idx[x] for x in df_test.label])

    X_train = load_images(df_train.path.tolist(), size=IMG_SIZE)
    X_test = load_images(df_test.path.tolist(), size=IMG_SIZE)

    y_train = to_categorical(y_train_idx, num_classes=len(labels))
    y_test = to_categorical(y_test_idx, num_classes=len(labels))

    train_ds, val_ds, X_tr, y_tr = make_datasets(
        X_train,
        y_train,
        batch=BATCH,
        val_size=0.2,
        augment=True
    )

    model = build_mini_xception(
        num_classes=len(labels),
        input_shape=(64, 64, 1)
    )

    print("\n======================")
    print(" TRAINING MODEL: mini-xception")
    print("======================\n")

    model.summary()

    # Train
    train(
        model,
        train_ds,
        val_ds,
        X_tr,
        y_tr,
        epochs=EPOCHS,
        out_dir=ART
    )

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH)
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTEST RESULT: loss={loss:.4f}, acc={acc:.4f}\n")

    with open(os.path.join(ART, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
