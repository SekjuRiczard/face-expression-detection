"""Mini-Xception architecture for Facial Expression Recognition.

This implementation uses separable convolutions, batch normalization,
PReLU activations and global average pooling. Based on the idea of the
Xception network, but heavily simplified for lightweight FER tasks.
"""

from tensorflow.keras import layers, models, losses, optimizers


def depthwise_separable(x, filters, name):
    """Apply a separable convolution block: SeparableConv2D + BatchNorm + PReLU.

       Args:
           x (Tensor): Input tensor.
           filters (int): Number of output channels.
           name (str): Prefix for layer names.

       Returns:
           Tensor: Output of separable convolution block.
       """
    x = layers.SeparableConv2D(filters, 3, padding="same",
                               name=name + "_sepconv")(x)
    x = layers.BatchNormalization(momentum=0.99, name=name + "_bn")(x)
    x = layers.PReLU(shared_axes=[1, 2], name=name + "_prelu")(x)
    return x


def build_mini_xception(num_classes, input_shape=(64, 64, 1), dropout=0.5):
    """Construct the mini-Xception CNN.

       Architecture:
           - 3 separable conv blocks with pooling
           - GlobalAveragePooling
           - Dropout
           - Dense softmax head

       Args:
           num_classes (int): Number of output categories.
           input_shape (tuple[int, int, int]): Input size.
           dropout (float): Dropout rate before output layer.

       Returns:
           tf.keras.Model: Compiled model ready for training.
       """
    inp = layers.Input(shape=input_shape)

    x = depthwise_separable(inp, 32, "b1")
    x = depthwise_separable(x, 32, "b1b")
    x = layers.MaxPooling2D(2)(x)

    x = depthwise_separable(x, 64, "b2")
    x = depthwise_separable(x, 64, "b2b")
    x = layers.MaxPooling2D(2)(x)

    x = depthwise_separable(x, 128, "b3")
    x = depthwise_separable(x, 128, "b3b")
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.2),
        metrics=["accuracy"]
    )

    return model
