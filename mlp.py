import jax
from jax import random, numpy as jnp, tree, vmap, nn
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Sequence


def init_params(key, dims: Sequence[int]):
    params = dict()
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        key, key2 = random.split(key, 2)
        W = random.normal(key, (in_dim, out_dim))
        b = random.normal(key2, (out_dim,))
        params[f'layer_{i + 1}'] = {'W': W, 'b': b}
    return params


@jax.jit
def forward_and_loss(params, x_batched, y_batched, eps=1e-15):
    def for_single_instance(x, y):
        for layer in list(params.values())[:-1]:
            x = jnp.dot(x, layer['W']) + layer['b']
            x = nn.relu(x)
        output_layer = list(params.values())[-1]
        x = jnp.dot(x, output_layer['W']) + output_layer['b']
        y_pred = nn.softmax(x)
        y_pred = jnp.clip(y_pred, eps, 1. - eps)
        loss = jnp.sum(y * jnp.log(y_pred))
        return loss
    return -jnp.mean(vmap(
        for_single_instance, in_axes=(0, 0))(x_batched, y_batched))


@jax.jit
def update_weights(params, grad, lr):
    return tree.map(lambda p, g: p - lr * g, params, grad)


batch_size = 512
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)
mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(
    lambda x, y: (tf.cast(tf.reshape(x, (batch_size, -1)), tf.float32) / 255.0,
                  tf.one_hot(y, depth=10)))


params = init_params(random.key(98765), (784, 256, 128, 64, 10))
grad_fn = jax.value_and_grad(forward_and_loss)
n_epochs = 100
n_batches = 0
lr = 0.01
per_epoch_losses = []
for epoch in range(1, n_epochs + 1):
    for batch, (x, y) in enumerate(mnist_ds, start=1):
        x = x.numpy()
        y = y.numpy()
        if epoch == 1:
            n_batches += 1
            batch_str = f"{batch}/?"
        else:
            batch_str = f"{batch}/{n_batches}"
        loss, grad = grad_fn(params, x, y)
        params = update_weights(params, grad, lr)
        print(f"\rEpoch: {epoch}/{n_epochs} | Batch {batch_str} | "
              f"Loss: {loss:.5f}", end="")
        per_epoch_losses.append(loss)
    print(f"\rEpoch: {epoch}/{n_epochs} | "
          f"Average Loss: {jnp.mean(jnp.array(per_epoch_losses)):.5f}", end="")
    print()
