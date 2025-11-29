"""Training utilities for FER models: dataset preparation, augmentation,
class balancing, learning-rate scheduling, and full model training routine.

Implements heavy augmentations such as MixUp, Cutout, random flipping,
translation, zoom, contrast and Gaussian noise.

Also constructs class-balanced training via sklearn.class_weight.
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def cutout(x, count=1, min_frac=0.1, max_frac=0.2):
    """Apply cutout augmentation by masking random square regions.

       Args:
           x (Tensor): Image tensor (B, H, W, C) or (H, W, C).
           count (int): Number of cutout regions.
           min_frac (float): Minimum cutout fraction of height.
           max_frac (float): Maximum cutout fraction of height.

       Returns:
           Tensor: Image tensor with cutout applied.
       """
    x_dtype = x.dtype
    x = tf.cast(x, tf.float32)

    h = tf.shape(x)[1]
    w = tf.shape(x)[2]

    for _ in range(count):
        size = tf.cast(tf.random.uniform([], min_frac, max_frac) * tf.cast(h, tf.float32), tf.int32)
        cy = tf.random.uniform([], 0, h, dtype=tf.int32)
        cx = tf.random.uniform([], 0, w, dtype=tf.int32)

        y1 = tf.clip_by_value(cy - size // 2, 0, h)
        y2 = tf.clip_by_value(cy + size // 2, 0, h)
        x1 = tf.clip_by_value(cx - size // 2, 0, w)
        x2 = tf.clip_by_value(cx + size // 2, 0, w)

        mask = tf.ones_like(x)
        mask = tf.tensor_scatter_nd_update(mask, [[0, y1, x1, 0]], [0.0])
        x = x * mask

    return tf.cast(x, x_dtype)


def mixup(batch_x, batch_y, alpha=0.3):
    """Apply MixUp augmentation between random sample pairs.

       Args:
           batch_x (Tensor): Batch of images.
           batch_y (Tensor): Batch of one-hot labels.
           alpha (float): MixUp alpha distribution parameter.

       Returns:
           tuple[Tensor, Tensor]: Mixed images and labels.
       """
    batch_x = tf.cast(batch_x, tf.float32)
    batch_y = tf.cast(batch_y, tf.float32)

    bs = tf.shape(batch_x)[0]

    if batch_x.shape[0] == 1:
        return batch_x, batch_y

    lam = tf.random.uniform([], 0.0, 1.0)
    lam = tf.maximum(lam, 1.0 - lam)

    idx = tf.random.shuffle(tf.range(bs))

    x1 = batch_x
    x2 = tf.gather(batch_x, idx)

    y1 = batch_y
    y2 = tf.gather(batch_y, idx)

    mx = lam * x1 + (1.0 - lam) * x2
    my = lam * y1 + (1.0 - lam) * y2

    return tf.cast(mx, batch_x.dtype), tf.cast(my, batch_y.dtype)


def make_datasets(X, y, batch=64, val_size=0.2, seed=42, augment=True):
    """Create shuffled TensorFlow datasets with optional augmentation.

        Args:
            X (np.ndarray): Raw images.
            y (np.ndarray): One-hot labels.
            batch (int): Batch size.
            val_size (float): Fraction of data reserved for validation.
            seed (int): Random seed for splitting.
            augment (bool): Whether to apply augmentation pipeline.

        Returns:
            tuple: (train_ds, val_ds, X_tr, y_tr)
        """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_size,
        stratify=np.argmax(y, axis=1),
        random_state=seed
    )

    def ds_from(Xd, yd, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((Xd, yd))
        if shuffle:
            ds = ds.shuffle(len(Xd), seed=seed)
        ds = ds.batch(batch)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = ds_from(X_tr, y_tr, shuffle=True)
    val_ds = ds_from(X_val, y_val)

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.GaussianNoise(0.03),
        ])

        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.map(lambda x, y: (cutout(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)


    return train_ds, val_ds, X_tr, y_tr


def get_callbacks(out_dir, base_lr=3e-4):
    """Create a list of Keras callbacks: warmup LR scheduler, checkpointing,
        early stopping, and ReduceLROnPlateau.

        Args:
            out_dir (str): Directory to store checkpoints.
            base_lr (float): Final learning rate.

        Returns:
            list[tf.keras.callbacks.Callback]: Configured callbacks.
        """
    def warmup(epoch):
        if epoch < 10:
            return 1e-5 + (base_lr - 1e-5) * (epoch / 10)
        return base_lr

    warmup_cb = tf.keras.callbacks.LearningRateScheduler(warmup)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(out_dir, "best.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        mode="max",
        restore_best_weights=True
    )

    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=6,
        min_lr=1e-5,
        mode="max",
        verbose=1
    )

    return [warmup_cb, ckpt, early, reduce]


def train(model, train_ds, val_ds, X_tr, y_tr, epochs=150, out_dir="../artifacts"):
    """Train a Keras model with balanced classes and standard callbacks.

        Args:
            model (tf.keras.Model): Model to train.
            train_ds (tf.data.Dataset): Training dataset.
            val_ds (tf.data.Dataset): Validation dataset.
            X_tr (np.ndarray): Raw training images for class weight computation.
            y_tr (np.ndarray): Raw training labels (one-hot).
            epochs (int): Number of epochs.
            out_dir (str): Output path for results.

        Returns:
            tf.keras.callbacks.History: Training history object.
        """
    os.makedirs(out_dir, exist_ok=True)

    y_idx = np.argmax(y_tr, axis=1)
    cw = compute_class_weight("balanced", classes=np.unique(y_idx), y=y_idx)
    class_weights = {i: float(cw[i]) for i in range(len(cw))}

    callbacks = get_callbacks(out_dir)

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save(os.path.join(out_dir, "last.keras"))
    return hist
