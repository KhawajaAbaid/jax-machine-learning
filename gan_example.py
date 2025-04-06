import jax
from jax import random, numpy as jnp, nn
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
from mlp import init_mlp_params
from optimizers import build_adam
from gan import train_step, generator_forward
from functools import partial


# =================================
# Initialize model parameters
# =================================
key = random.key(1999)
latent_dim = 100
image_size = 28 * 28
generator_params = init_mlp_params(
    key,
    (latent_dim, 256, 512, 1024, image_size))

key = random.split(key, 1)[0]
discriminator_params = init_mlp_params(key, (image_size, 512, 256, 128, 1))

generator_optimizer_params = build_adam(generator_params)
discriminator_optimizer_params = build_adam(discriminator_params)


# =================================
# Load MNIST dataset
# =================================
batch_size = 2048
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)

mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(
    lambda x, y: (tf.cast(tf.reshape(x, (batch_size, -1)),
                          tf.float32) - 127.5) / 127.5)


# ==============================================
# Utility function to generate and save images
# ==============================================
def generate_and_save_image(generator_params, epoch, key,
                            hidden_activation_fn: callable,
                            output_activation_fn: callable):
    noise = random.normal(key, (1, latent_dim))
    fake_image = generator_forward(generator_params, noise,
                                   hidden_activation_fn=hidden_activation_fn,
                                   output_activation_fn=output_activation_fn)
    fake_image = np.asarray(fake_image).reshape((28, 28)) * 127.5 + 127.5
    img = Image.fromarray(fake_image).convert("L")
    directory_path = "./images-jax-gan"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    img.save(f"{directory_path}/epoch_{epoch}.png")


# ==================================
# Training loop
# ==================================
n_epochs = 1000
n_batches = 0
learning_rate = 0.0002
gen_per_batch_losses = []
disc_per_batch_losses = []
key = random.key(4444)
for epoch in range(1, n_epochs + 1):
    batch_str = "if you're seeing this, something's wrong"
    for batch, x in enumerate(mnist_ds, start=1):
        x = x.numpy()
        if epoch == 1:
            n_batches += 1
            batch_str = f"{batch}/?"
        else:
            batch_str = f"{batch}/{n_batches}"

        key, key_2 = random.split(key, 2)
        noise_batched_for_discriminator = random.normal(
            key,
            shape=(x.shape[0], latent_dim))
        noise_batched_for_generator = random.normal(
            key_2,
            shape=(x.shape[0], latent_dim))
        generator_params, discriminator_params, metrics = train_step(
            generator_params,
            discriminator_params,
            generator_optimizer_params,
            discriminator_optimizer_params,
            x_batched=x,
            noise_batched_for_generator=noise_batched_for_generator,
            noise_batched_for_discriminator=noise_batched_for_discriminator,
            batch_num=batch,
            generator_hidden_activation_fn=partial(nn.leaky_relu, negative_slope=0.2),
            generator_output_activation_fn=nn.tanh,
            discriminator_hidden_activation_fn=partial(nn.leaky_relu, negative_slope=0.2),
            learning_rate=learning_rate,
        )

        print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
              f"Generator Loss: {metrics['generator_loss']:.6f} | "
              f"Discriminator Loss: {metrics['discriminator_loss']:.6f}",
              end="")
        gen_per_batch_losses.append(metrics['generator_loss'])
        disc_per_batch_losses.append(metrics['discriminator_loss'])
    generator_loss = jnp.mean(jnp.asarray(gen_per_batch_losses))
    discriminator_loss = jnp.mean(jnp.asarray(disc_per_batch_losses))
    print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
          f"Generator_loss Loss: {generator_loss:.6f} | "
          f"Discriminator Loss: {discriminator_loss:.6f}",
          end="")
    key = random.split(key, 1)[0]
    if epoch % 10 == 0:
        generate_and_save_image(
            generator_params,
            epoch,
            key,
            partial(nn.leaky_relu, negative_slope=0.2),
            nn.tanh)
    print()

