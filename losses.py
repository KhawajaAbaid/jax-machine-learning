from jax import numpy as jnp


def cross_entropy_loss(
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        eps: float = 1e-15
        ) -> jnp.ndarray:
    """
    Computes the log loss / cross entropy loss between true and predicted values.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        eps: Small value to avoid log(0).
    Returns:
        Log loss value.
    """
    y_pred = jnp.clip(y_pred, eps, 1. - eps)
    loss = - jnp.sum(y_true * jnp.log(y_pred), axis=-1)
    return loss
