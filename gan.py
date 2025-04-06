import jax
from jax import numpy as jnp, vmap, tree, nn
from mlp import MLP
from optimizers import apply_adam
from functools import partial



def generator_forward(
        params: MLP,
        x: jnp.ndarray,
        hidden_activation_fn: callable,
        output_activation_fn: callable,
        ):
    for layer in params.layers[:-1]:
        x = jnp.matmul(x, layer.w) + layer.b
        x = hidden_activation_fn(x)
    last_layer = params.layers[-1]
    logits = jnp.matmul(x, last_layer.w) + last_layer.b
    image = output_activation_fn(logits)
    return image

def generator_forward_and_loss(
        generator_params: MLP,
        discriminator_params: MLP,
        noise_batched: jnp.ndarray,
        hidden_activation_fn: callable,
        output_activation_fn: callable,
        discriminaor_forward: callable,
        ):
    def for_single_instance(noise):
        fake_image = generator_forward(
            generator_params,
            noise,
            hidden_activation_fn,
            output_activation_fn
            )
        logits = discriminaor_forward(discriminator_params, fake_image)
        # binary cross entropy loss - ground truth for every fake image is 1 here
        # because we want to fool the discriminator
        loss = -1.0 * nn.log_sigmoid(logits)
        return loss
    return jnp.mean(
        vmap(for_single_instance, in_axes=0)(noise_batched)
    )

def discriminaor_forward(
        params: MLP,
        x: jnp.ndarray,
        hidden_activation_fn: callable,
        ):
    for layer in params.layers[:-1]:
        x = jnp.matmul(x, layer.w) + layer.b
        x = hidden_activation_fn(x)
    last_layer = params.layers[-1]
    logits = jnp.matmul(x, last_layer.w) + last_layer.b
    return logits


def discriminator_forward_and_loss(
        params: MLP,
        x_batched: jnp.ndarray,
        y_batched: jnp.ndarray,
        hidden_activation_fn: callable,
        ):
    def for_single_instance(x, y):
        logits = discriminaor_forward(
            params,
            x,
            hidden_activation_fn
            )
        # binary cross entropy loss
        loss = -y * nn.log_sigmoid(logits) - (1.0 - y) * nn.log_sigmoid(-logits)
        return loss
    return jnp.mean(
        vmap(for_single_instance, in_axes=(0, 0))(x_batched, y_batched)
    )


discriminator_grad_fn = jax.value_and_grad(discriminator_forward_and_loss)
generator_grad_fn = jax.value_and_grad(generator_forward_and_loss)


@partial(
        jax.jit,
        static_argnames=(
            'generator_hidden_activation_fn',
            'generator_output_activation_fn',
            'discriminator_hidden_activation_fn',
            'learning_rate'
            )
)
def train_step(
        generator_params: MLP,
        discriminator_params: MLP,
        generator_optimizer_params: dict,
        discriminator_optimizer_params: dict,
        x_batched: jnp.ndarray,
        noise_batched_for_generator: jnp.ndarray,
        noise_batched_for_discriminator: jnp.ndarray,
        batch_num: int,
        generator_hidden_activation_fn: callable,
        generator_output_activation_fn: callable,
        discriminator_hidden_activation_fn: callable,
        learning_rate: float = 0.0002,

):
    """
    Train step for GAN.

    Args:
        generator_params: Generator parameters.
        discriminator_params: Discriminator parameters.
        generator_optimizer_params: Generator optimizer parameters.
        discriminator_optimizer_params: Discriminator optimizer parameters.
        x_batched: Real images.
        noise_batched_for_generator: Noise for generator train step.
        noise_batched_for_discriminator: Noise for discriminator train step.
        batch_num: Current batch number.
        generator_hidden_activation_fn: Generator hidden activation function.
        generator_output_activation_fn: Generator output activation function.
        discriminator_hidden_activation_fn: Discriminator hidden activation function.
        learning_rate: Learning rate.
    """
    metrics = dict()
    # =========================
    # Discriminator train step
    # =========================
    discriminator_loss_real, discriminator_grads_real = discriminator_grad_fn(
        discriminator_params,
        x_batched,
        jnp.ones(shape=(x_batched.shape[0], 1)),
        hidden_activation_fn=discriminator_hidden_activation_fn
    )
    fake_images = vmap(generator_forward, in_axes=(None, 0, None, None))(
        generator_params,
        noise_batched_for_discriminator,
        generator_hidden_activation_fn,
        generator_output_activation_fn
    )
    discriminator_loss_fake, discriminator_grads_fake = discriminator_grad_fn(
        discriminator_params,
        fake_images,
        jnp.zeros(shape=(x_batched.shape[0], 1)),
        hidden_activation_fn=discriminator_hidden_activation_fn
    )
    discriminator_grads = tree.map(
        lambda g1, g2: g1 + g2,
        discriminator_grads_real,
        discriminator_grads_fake
    )

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
    discriminaor_forward_fn = partial(discriminaor_forward,
                                      hidden_activation_fn=discriminator_hidden_activation_fn)
    generator_loss, generator_grads = generator_grad_fn(
        generator_params,
        discriminator_params,
        noise_batched_for_generator,
        generator_hidden_activation_fn,
        generator_output_activation_fn,
        discriminaor_forward_fn,
    )

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

