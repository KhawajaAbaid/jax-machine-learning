import jax
from jax import random, numpy as jnp, tree, vmap, nn
from mlp import init_mlp_params, forward_and_loss
from losses import cross_entropy_loss
import tensorflow as tf
import tensorflow_datasets as tfds
from optimizers import apply_sgd

# =================================
# Load MNIST dataset
# =================================
batch_size = 512
mnist_ds = tfds.load('mnist', split='train', shuffle_files=True,
                     as_supervised=True)
mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)
mnist_ds = mnist_ds.map(
    lambda x, y: (tf.cast(tf.reshape(x, (batch_size, -1)), tf.float32) / 255.0,
                  tf.one_hot(y, depth=10)))


# =================================
# Initialize model parameters
# =================================
params = init_mlp_params(random.key(98765), (784, 256, 128, 64, 10))
grad_fn = jax.value_and_grad(forward_and_loss)
apply_sgd_jitted = jax.jit(apply_sgd)
n_epochs = 100
n_batches = 0
lr = 0.01

# =================================
# Training loop
# =================================
per_epoch_losses = []
for epoch in range(1, n_epochs + 1):
    for batch, (x, y) in enumerate(mnist_ds, start=1):
        x = x.numpy()
        y = y.numpy()
        if epoch == 1:
            n_batches += 1
            batch_str = f"{batch}/?  "
        else:
            batch_str = f"{batch:3}/{n_batches:3}"
        loss, grad = grad_fn(params, x, y,
                             hidden_activation_fn=nn.relu,
                             output_activation_fn=nn.softmax,
                             loss_fn=cross_entropy_loss)
        params = apply_sgd_jitted(params, grad, lr)
        print(f"\rEpoch: {epoch:3}/{n_epochs:3} | Batch {batch_str} | "
              f"Loss: {loss:8.5f}", end="")
        per_epoch_losses.append(loss)
    print(f" | Average Loss: {jnp.mean(jnp.array(per_epoch_losses)):8.5f}", end="")
    print()
