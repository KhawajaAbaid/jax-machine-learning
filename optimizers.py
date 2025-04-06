import jax
from jax import tree, numpy as jnp


# =================================
# SGD Optimizer
# =================================
def apply_sgd(params, grad, lr):
    return tree.map(lambda p, g: p - lr * g, params, grad)



# =================================
# ADAM Optimizer
# =================================
def build_adam(
        params
):
    m = tree.map(lambda x: jnp.zeros(jnp.shape(x)), params)
    v = tree.map(lambda x: jnp.zeros(jnp.shape(x)), params)
    return {'m': m, 'v': v}


def apply_adam(
        adam_params,
        model_params,
        grads,
        batch_num,
        learning_rate=0.001,
        b1=0.9,
        b2=0.999,
        epsilon=10e-8,
):
    adam_params['m'] = tree.map(
        lambda m, g: b1 * m + (1 - b1) * g,
        adam_params['m'],
        grads
    )
    adam_params['v'] = tree.map(
        lambda v, g: b2 * v + (1 - b2) * jnp.square(g),
        adam_params['v'],
        grads
    )

    alpha_t = learning_rate * (
            jnp.sqrt(1.0 - jnp.pow(b2, batch_num)) / (1 - jnp.pow(b1, batch_num)))
    model_params = tree.map(
        lambda p, g: p - alpha_t * g,
        model_params,
        tree.map(
            lambda m, v: m / (jnp.sqrt(v) + epsilon),
            adam_params['m'],
            adam_params['v']
        )
    )
    return model_params, adam_params

