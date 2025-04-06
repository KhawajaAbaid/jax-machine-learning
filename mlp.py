import jax
from jax import random, numpy as jnp, nn, tree_util, vmap
from typing import Sequence, Union
from types_util import ActivationType
from dtypes import default_dtype
from functools import partial


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


@partial(jax.jit, static_argnames=('hidden_activation_fn', 'output_activation_fn', 'loss_fn'))
def forward_and_loss(
    params,
    x_batched,
    y_batched,
    hidden_activation_fn: callable,
    output_activation_fn: callable,
    loss_fn: callable,
    ):
    def for_single_instance(x, y):
        for layer in params.layers[:-1]:
            x = jnp.matmul(x, layer.w)
            if layer.b is not None:
                x = x + layer.b
            x = hidden_activation_fn(x)
        output_layer = params.layers[-1]
        x = jnp.matmul(x, output_layer.w)
        if output_layer.b is not None:
            x = x + output_layer.b
        # Apply outptut activation and compute loss
        y_pred = output_activation_fn(x)
        loss = loss_fn(y, y_pred)
        return loss
    return jnp.mean(vmap(
        for_single_instance, in_axes=(0, 0))(x_batched, y_batched))