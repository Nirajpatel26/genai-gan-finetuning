"""
DCGAN Training Loop with Stability Techniques
GenAI Course — Spring 2026 | Northeastern University

Key stability features:
- Label smoothing (real labels → 0.9)
- Asymmetric learning rates (G: 2e-4, D: 1e-4)
- Adam optimizer with beta_1=0.5 (DCGAN standard)
- Periodic sample saving for visual monitoring
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models.generator import build_generator
from src.models.discriminator import build_discriminator

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM    = 100
EPOCHS        = 200
BATCH_SIZE    = 64
G_LR          = 2e-4
D_LR          = 1e-4
LABEL_SMOOTH  = 0.9   # Smooth real labels to prevent discriminator overconfidence
SAVE_INTERVAL = 10    # Save generated samples every N epochs

# ── Loss & Optimizers ────────────────────────────────────────────────────────
cross_entropy = tf.keras.losses.BinaryCrossentropy()

g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.5)


def discriminator_loss(real_output, fake_output, label_smooth=LABEL_SMOOTH):
    """
    D wants: real → 1 (smoothed), fake → 0
    Label smoothing prevents D from becoming too confident,
    which would starve G of useful gradients.
    """
    real_loss = cross_entropy(tf.ones_like(real_output) * label_smooth, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    """
    G wants: fake → 1 (fool D into thinking they're real)
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(real_images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    return g_loss, d_loss


def save_samples(generator, epoch, seed, save_dir="results/images"):
    os.makedirs(save_dir, exist_ok=True)
    predictions = generator(seed, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        img = (predictions[i].numpy() * 0.5 + 0.5)  # Rescale [-1,1] → [0,1]
        ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:03d}.png")
    plt.close()


def train(dataset):
    generator     = build_generator(LATENT_DIM)
    discriminator = build_discriminator()
    seed          = tf.random.normal([16, LATENT_DIM])  # Fixed seed for visual comparison

    g_losses, d_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        epoch_g, epoch_d = [], []
        for batch in dataset:
            g_loss, d_loss = train_step(batch, generator, discriminator)
            epoch_g.append(g_loss.numpy())
            epoch_d.append(d_loss.numpy())

        mean_g = np.mean(epoch_g)
        mean_d = np.mean(epoch_d)
        g_losses.append(mean_g)
        d_losses.append(mean_d)

        print(f"Epoch {epoch:3d}/{EPOCHS} | G Loss: {mean_g:.4f} | D Loss: {mean_d:.4f}")

        if epoch % SAVE_INTERVAL == 0:
            save_samples(generator, epoch, seed)

    # Save final models
    generator.save("results/generator_final.keras")
    discriminator.save("results/discriminator_final.keras")

    # Save loss curves
    _plot_losses(g_losses, d_losses)
    return generator, discriminator


def _plot_losses(g_losses, d_losses):
    os.makedirs("results/loss_curves", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", color="blue")
    plt.plot(d_losses, label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DCGAN Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/loss_curves/training_loss.png")
    plt.close()
    print("Loss curves saved.")
