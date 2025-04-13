import jax
from jax import numpy as jnp, random, nn, tree_util, vmap
from typing import Sequence


@tree_util.register_pytree_node_class
class Conv2D:
    def __init__(
            self,
            num_filters: int,
            kernel: jnp.ndarray,
            bias: jnp.ndarray = None,
            strides: Sequence[int] = (1, 1),
            padding: str = 'VALID',
    ):
        self.num_filters = num_filters
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
    
    def tree_flatten(self):
        children = (
            self.num_filters,
            self.kernel,
            self.bias,
            self.strides,
            self.padding
            )
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def __repr__(self):
        return f"Conv2D(kernel={self.kernel.shape}, bias={self.bias.shape if self.bias is not None else None})"


@tree_util.register_pytree_node_class
class CNN:
    def __init__(self, layers: Sequence[Conv2D]):
        self.layers = layers
    
    def add(self, layer: Conv2D):
        self.layers.append(layer)
    
    def tree_flatten(self):
        children = self.layers
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)
    
    def __repr__(self):
        return f"CNN(layers={len(self.layers)})"


def init_conv2d_params(
        key,
        num_filters: int,
        kernel_shape: Sequence[int],
        input_shape: Sequence[int],
        initializer: callable = nn.initializers.glorot_normal(),
        use_bias: bool = True,
        strides: Sequence[int] = (1, 1),
        padding: str = 'VALID'
):
    """
    Initialize the parameters for a Conv2D layer.
    
    Args:
        key: Random key for initialization.
        num_filters: Number of filters in the Conv2D layer.
        kernel_shape: Shape of the kernel (height, width).
        input_shape: Shape of the input (height, width, channels).
        initializer: Function to initialize the weights.
        use_bias: Whether to use bias or not.
        strides: Strides for the convolution.
        padding: Padding type ('VALID' or 'SAME').
    """
    num_channels = input_shape[-1]
    kernel_key, bias_key = random.split(key, 2)
    kernel = initializer(kernel_key, (kernel_shape[0], kernel_shape[1], num_channels, num_filters))
    bias = None
    if use_bias:
        bias = initializer(bias_key, (num_filters,))
    return Conv2D(num_filters, kernel, bias, strides, padding)


def init_cnn_params(
        key,
        input_shape: Sequence[int],
        layer_configs: Sequence[dict],
        initializer: callable = nn.initializers.glorot_normal(),
):
    """
    Initialize the parameters for a CNN.
    
    Args:
        key: Random key for initialization.
        input_shape: Shape of the input (height, width, channels).
        layer_configs: List of dictionaries containing Conv2D layer configurations.
        initializer: Function to initialize the weights.
    """
    layers = []
    for config in layer_configs:
        key, subkey = random.split(key)
        layer = init_conv2d_params(
            subkey,
            input_shape=input_shape,
            **config
        )
        layers.append(layer)
        input_shape = (
            (input_shape[0] - config['kernel_shape'][0]) // config['strides'][0] + 1,
            (input_shape[1] - config['kernel_shape'][1]) // config['strides'][1] + 1,
            config['num_filters']
        )
    return CNN(layers)