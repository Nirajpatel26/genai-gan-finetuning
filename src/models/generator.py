"""
DCGAN Generator Architecture
GenAI Course — Spring 2026 | Northeastern University
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_generator(latent_dim=100, output_channels=3):
    """
    DCGAN Generator: Latent vector → 64x64 RGB image
    Architecture follows DCGAN paper guidelines:
    - No fully-connected layers in conv part
    - BatchNormalization after every Conv (except output)
    - ReLU activations in generator, Tanh at output
    """
    model = tf.keras.Sequential(name="Generator")

    # Foundation: dense + reshape into spatial feature map
    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((4, 4, 512)))

    # Upsample: 4x4 → 8x8
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Upsample: 8x8 → 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Upsample: 16x16 → 32x32
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Upsample: 32x32 → 64x64 (output)
    model.add(layers.Conv2DTranspose(output_channels, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.Activation("tanh"))  # Output in [-1, 1]

    return model


if __name__ == "__main__":
    gen = build_generator()
    gen.summary()
    # Test forward pass
    noise = tf.random.normal([1, 100])
    output = gen(noise, training=False)
    print(f"Generator output shape: {output.shape}")  # (1, 64, 64, 3)
