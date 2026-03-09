"""
DCGAN Discriminator Architecture
GenAI Course — Spring 2026 | Northeastern University
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_discriminator(input_shape=(64, 64, 3)):
    """
    DCGAN Discriminator: 64x64 RGB image → real/fake probability
    Architecture follows DCGAN paper guidelines:
    - Strided convolutions (no pooling layers)
    - BatchNormalization after every Conv (except input)
    - LeakyReLU(0.2) activations throughout
    - Sigmoid output
    """
    model = tf.keras.Sequential(name="Discriminator")

    # Input: 64x64 → 32x32 (no BatchNorm on first layer — DCGAN guideline)
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", input_shape=input_shape))
    model.add(layers.LeakyReLU(0.2))

    # 32x32 → 16x16
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # 16x16 → 8x8
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # 8x8 → 4x4
    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    # Classify
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


if __name__ == "__main__":
    disc = build_discriminator()
    disc.summary()
    # Test forward pass
    img = tf.random.normal([1, 64, 64, 3])
    pred = disc(img, training=False)
    print(f"Discriminator output: {pred.numpy()}")  # Probability in [0, 1]
