import jax
from jax import random, numpy as jnp, tree


n_samples = 16
x_dim = 5
y_dim = 3

# True params
key = random.key(1337)
key1, key2 = random.split(key, 2)
W = random.normal(key1, (x_dim, y_dim))
b = random.normal(key2, (y_dim,))

# Generate data
key_samples, key_noise = random.split(key1)
x_samples = random.normal(key_samples, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(
    key_noise, (n_samples, y_dim))


@jax.jit
def forward_and_loss(params, x_batched, y_batched):
    def for_single_instance(x, y):
        y_pred = jnp.dot(x, params["W"]) + params["b"]
        loss = jnp.inner(y - y_pred, y - y_pred) / 2.0
        return loss
    return jnp.mean(jax.vmap(for_single_instance,
                             in_axes=(0, 0))(x_batched, y_batched))


key_approx_1, key_approx_2 = random.split(key2)
W_approx = random.normal(key_approx_1, W.shape)
b_approx = random.normal(key_approx_2, b.shape)
params = {"W": W_approx, "b": b_approx}


grad_fn = jax.value_and_grad(forward_and_loss)
lr = 0.01  # learning rate
for epoch in range(100):
    loss, grad = grad_fn(params, x_samples, y_samples)
    params = tree.map(lambda p, g: p - lr * g, params, grad)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}")
