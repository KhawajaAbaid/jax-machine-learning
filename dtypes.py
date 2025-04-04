from jax import numpy as jnp


_default_dtype = jnp.float32

def set_default_dtype(dtype):
    global _default_dtype
    _default_dtype = dtype

def default_dtype():
    return _default_dtype