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
        bias = initializer(bias_key, (num_filters, 1))
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

def apply_conv2d(
        conv2d: Conv2D,
        x: jnp.ndarray,
):
    """
    Apply a Conv2D layer to an input tensor.
    
    Args:
        conv2d: Conv2D layer parameters.
        x: Input tensor of shape (batch_size, height, width, channels).
    
    Returns:
        Output tensor after applying the Conv2D layer.
    """
    batch_size, in_height, in_width, in_channels = x.shape
    output_shape = (
        (x.shape[1] - conv2d.kernel.shape[0]) // conv2d.strides[0] + 1,
        (x.shape[2] - conv2d.kernel.shape[1]) // conv2d.strides[1] + 1,
        conv2d.num_filters
    )
    output = jnp.zeros((x.shape[0], *output_shape), dtype=x.dtype, device=x.device)
    for c in range(conv2d.num_filters):
        kernel = conv2d.kernel[..., c]
        bias = conv2d.bias[c]
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                h_start = i * conv2d.strides[0]     # horizontal start
                h_end = h_start + conv2d.kernel.shape[0]
                v_start = j * conv2d.strides[1]    # vertical start
                v_end = v_start + conv2d.kernel.shape[1]
                region = x[:, v_start:v_end, h_start:h_end, :]
                prod = region * kernel
                summed = jnp.sum(prod, axis=(1, 2, 3))
                final = summed + bias
                output = output.at[:, i, j, c].set(final)
    return output


def apply_cnn(
        cnn: CNN,
        x: jnp.ndarray,
):
    """
    Apply a CNN to an input tensor.
    
    Args:
        cnn: CNN parameters.
        x: Input tensor of shape (batch_size, height, width, channels).
    
    Returns:
        Output tensor after applying the CNN.
    """
    for layer in cnn.layers:
        x = apply_conv2d(layer, x)
    return x