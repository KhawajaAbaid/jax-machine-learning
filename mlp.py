import jax
from jax import random, numpy as jnp, tree, vmap, nn, tree_util
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Sequence, Union
from types_util import ActivationType
from dtypes import default_dtype


@tree_util.register_pytree_node_class
class Linear:
    """"
    A PyTree holding parameters for the linear layer.
    Please do not treat this as a layer. 
    This is just a container for the parameters.

    Args:
        w: The weight matrix of the layer.
        b: The bias vector of the layer.
    """
    def __init__(
            self,
            w: jnp.ndarray,
            b: jnp.ndarray = None,
            ):
        self.w = w
        self.b = b
    
    def tree_flatten(self):
        children = (self.w, self.b)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        return f"Linear(w={self.w.shape}, b={self.b.shape if self.b is not None else None})"


@tree_util.register_pytree_node_class
class MLP:
    """
    A PyTree holding parameters for the MLP.
    Please do not treat this as a model.
    This is just a container for the parameters.

    Args:
        layers: A list of Linear layers pytrees.
    """
    def __init__(self, layers: Sequence[Linear]):
        self.layers = layers
    
    def add(self, layer: Linear):
        self.layers.append(layer)

    def tree_flatten(self):
        children = self.layers
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)


def init_linear_params(
        key,
        input_dims: int,
        output_dims: int,
        use_bias: bool = True,
        initializer=nn.initializers.glorot_normal()
        ):
    w_key, b_key = random.split(key, 2)
    w = initializer(w_key, (input_dims, output_dims))
    b = None
    if use_bias:
        b = initializer(b_key, (1, output_dims))
    return Linear(w, b)


def init_mlp_params(
        key,
        dims: Sequence[int],
        use_bias: bool = True,
        initializer=nn.initializers.glorot_normal()
        ):
    layers = []
    for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
        key, subkey = random.split(key, 2)
        layer = init_linear_params(subkey, input_dim, output_dim, use_bias, initializer)
        layers.append(layer)
    return MLP(layers)


@jax.jit
def forward_and_loss(params, x_batched, y_batched, eps=1e-15):
    def for_single_instance(x, y):
        for layer in params.layers[:-1]:
            x = jnp.matmul(x, layer.w)
            if layer.b is not None:
                x = x + layer.b
            x = nn.relu(x)
        output_layer = params.layers[-1]
        x = jnp.matmul(x, output_layer.w)
        if output_layer.b is not None:
            x = x + output_layer.b
        # Apply softmax and compute loss
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


params = init_mlp_params(random.key(98765), (784, 256, 128, 64, 10))
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
            batch_str = f"{batch}/?  "
        else:
            batch_str = f"{batch:3}/{n_batches:3}"
        loss, grad = grad_fn(params, x, y)
        params = update_weights(params, grad, lr)
        print(f"\rEpoch: {epoch:3}/{n_epochs:3} | Batch {batch_str} | "
              f"Loss: {loss:8.5f}", end="")
        per_epoch_losses.append(loss)
    print(f" | Average Loss: {jnp.mean(jnp.array(per_epoch_losses)):8.5f}", end="")
    print()
