from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(
    input_shape: Tuple[int, int, int] = (64, 64, 1),
    base_filters: int = 32,
    latent_dim: int = 128,
) -> tf.keras.Model:
    """
    Build and compile a simple convolutional autoencoder.

    Args:
        input_shape: (H, W, C), e.g. (64, 64, 1) for grayscale frames.
        base_filters: number of filters for the first conv layer.
        latent_dim: currently unused explicitly, but kept for future extensions.

    Returns:
        A compiled tf.keras.Model with MSE loss and Adam optimizer.
    """
    inputs = layers.Input(shape=input_shape, name="input_frame")

    # Encoder
    x = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D(2, padding="same")(x)  # 64 -> 32

    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(2, padding="same")(x)  # 32 -> 16

    x = layers.Conv2D(base_filters * 4, 3, padding="same", activation="relu")(x)
    encoded = layers.MaxPool2D(2, padding="same", name="encoded")(x)  # 16 -> 8

    # Decoder
    x = layers.Conv2D(base_filters * 4, 3, padding="same", activation="relu")(encoded)
    x = layers.UpSampling2D(2)(x)  # 8 -> 16

    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2)(x)  # 16 -> 32

    x = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2)(x)  # 32 -> 64

    outputs = layers.Conv2D(
        input_shape[-1],
        3,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(x)

    model = models.Model(inputs, outputs, name="conv_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model
