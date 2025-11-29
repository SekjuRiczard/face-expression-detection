"""Utility functions for model evaluation, including logits prediction,
classification reports, and confusion matrix generation.

This module provides helpers for evaluating trained TensorFlow/Keras models
on in-memory numpy datasets. It does not load images from disk; instead, it
expects already preprocessed arrays.

Dependencies:
    - numpy
    - tensorflow
    - scikit-learn (confusion_matrix, classification_report)
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def predict_logits(model, X, batch=64):
    """Run forward-pass prediction and return raw model logits.

       Args:
           model (tf.keras.Model): Loaded model used for inference.
           X (np.ndarray): Input data of shape (N, H, W, C).
           batch (int): Batch size for Dataset batching.

       Returns:
           np.ndarray: Model outputs (logits or probabilities), shape (N, num_classes).
       """
    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch)
    return model.predict(ds)

def evaluate(model, X, y_true, class_names):
    """Compute classification metrics: report + confusion matrix.

       Args:
           model (tf.keras.Model): Trained model.
           X (np.ndarray): Input samples.
           y_true (np.ndarray): One-hot encoded true labels.
           class_names (list[str]): Names of classes in correct index order.

       Returns:
           np.ndarray: Confusion matrix of shape (C, C).
       """
    logits = predict_logits(model, X)
    y_pred = np.argmax(logits, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)

    print("\nClassification report:")
    print(classification_report(y_true_idx, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true_idx, y_pred)
    print("\nConfusion matrix:\n", cm)
    return cm
