import jax
from jax import random, numpy as jnp, vmap, tree, nn
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Sequence
import numpy as np
from PIL import Image
import os
from optimizers import build_adam, apply_adam


def init_mlp_params(
        key,
        dims: Sequence[int],
        initializer=nn.initializers.glorot_normal()):
    params = dict()
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        key, key2 = random.split(key, 2)
        W = initializer(key, (in_dim, out_dim), jnp.float32)
        b = initializer(key2, (1, out_dim), jnp.float32)
        params[f'layer_{i + 1}'] = {'W': W, 'b': b}
    return params


def generator_forward(params, x):
    for layer in list(params.values())[:-1]:
        x = jnp.dot(x, layer['W']) + layer['b']
        x = nn.leaky_relu(x, negative_slope=0.2)
    last_layer = list(params.values())[-1]
    logits = jnp.dot(x, last_layer['W']) + last_layer['b']
    image = nn.tanh(logits)
    return image


def discriminaor_forward(params, x):
    for layer in list(params.values())[:-1]:
        x = jnp.dot(x, layer['W']) + layer['b']
        x = nn.leaky_relu(x, negative_slope=0.2)
    last_layer = list(params.values())[-1]
    logits = jnp.dot(x, last_layer['W']) + last_layer['b']
    return logits


def discriminator_forward_and_loss(params, x_batched, y_batched):
    def for_single_instance(x, y):
        logits = discriminaor_forward(params, x)
        # binary cross entropy loss
        loss = -y * nn.log_sigmoid(logits) - (1.0 - y) * nn.log_sigmoid(-logits)
        return loss
    return jnp.mean(
        vmap(for_single_instance, in_axes=(0, 0))(x_batched, y_batched)
    )


def generator_forward_and_loss(
        generator_params,
        discriminator_params,
        noise_batched):
    def for_single_instance(noise):
        fake_image = generator_forward(generator_params, noise)
        logits = discriminaor_forward(discriminator_params, fake_image)
        # binary cross entropy loss
        loss = -1.0 * nn.log_sigmoid(logits)
        return loss
    return jnp.mean(
        vmap(for_single_instance, in_axes=0)(noise_batched)
    )


discriminator_grad_fn = jax.value_and_grad(discriminator_forward_and_loss)
generator_grad_fn = jax.value_and_grad(generator_forward_and_loss)


@jax.jit
def train_step(
        generator_params,
        discriminator_params,
        generator_optimizer_params,
        discriminator_optimizer_params,
        batch_num,
        x_batched,
        noise_batched_for_discriminator,
        noise_batched_for_generator,
        learning_rate=0.0002,
):
    """
    noise_batched_for_discriminator means that the noise will be used in the
    TRAIN STEP of the discriminator NOT BY THE discriminator itself.
    if you find it confusing, well, sorry.
    """
    metrics = dict()
    # ========================
    # Discriminator train step
    # ========================
    discriminator_loss_real, discriminator_grads_real = discriminator_grad_fn(
        discriminator_params,
        x_batched,
        jnp.ones(shape=(x_batched.shape[0], 1))
    )
    fake_images = vmap(generator_forward, in_axes=(None, 0))(
        generator_params,
        noise_batched_for_discriminator
    )
    discriminator_loss_fake, discriminator_grads_fake = discriminator_grad_fn(
        discriminator_params,
        fake_images,
        jnp.zeros(shape=(x_batched.shape[0], 1))
    )
    discriminator_grads = tree.map(
        lambda g1, g2: g1 + g2,
        discriminator_grads_real,
        discriminator_grads_fake
    )
    # discriminator_params = tree.map(
    #     lambda p, g: p - learning_rate * g,
    #     discriminator_params,
    #     discriminator_grads
    # )

    discriminator_params, discriminator_optimizer_params = apply_adam(
        discriminator_optimizer_params,
        discriminator_params,
        discriminator_grads,
        batch_num,
        learning_rate,
        b1=0.5,
    )

    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    metrics.update({'discriminator_loss': discriminator_loss})

    # ======================
    # Generator train step
    # ======================
    generator_loss, generator_grads = generator_grad_fn(
        generator_params,
        discriminator_params,
        noise_batched_for_generator
    )
    # generator_params = tree.map(
    #     lambda p, g: p - learning_rate * g,
    #     generator_params,
    #     generator_grads
    # )
    generator_params, generator_optimizer_params = apply_adam(
        generator_optimizer_params,
        generator_params,
        generator_grads,
        batch_num,
        learning_rate,
        b1=0.5,
    )
    metrics.update({'generator_loss': generator_loss})
    return generator_params, discriminator_params, metrics


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


# Load dataset
batch_size = 2048
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)

mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(
    lambda x, y: (tf.cast(tf.reshape(x, (batch_size, -1)),
                          tf.float32) - 127.5) / 127.5)


# Utility function
def generate_and_save_image(generator_params, epoch, key):
    noise = random.normal(key, (1, latent_dim))
    fake_image = generator_forward(generator_params, noise)
    fake_image = np.asarray(fake_image).reshape((28, 28)) * 127.5 + 127.5
    img = Image.fromarray(fake_image).convert("L")
    directory_path = "./images-jax-gan"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    img.save(f"{directory_path}/epoch_{epoch}.png")


# Training loop
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
            batch_num=batch,
            noise_batched_for_generator=noise_batched_for_generator,
            noise_batched_for_discriminator=noise_batched_for_discriminator,
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
        generate_and_save_image(generator_params, epoch, key)
    print()

