# secure_gbdt/primitives.py
# This module abstracts all secure primitives.

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def add_dp_noise(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """
    Applies the Laplace mechanism for Differential Privacy.
    noise ~ Laplace(0, sensitivity / epsilon)
    """
    if epsilon <= 0.0:
        return value
    noise = np.random.laplace(0, sensitivity / epsilon)
    return value + noise

def greater_than(a, b):
    # TODO: secure greater-than (Fgreater)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return (a > b).astype(np.int8)
    if isinstance(a, np.ndarray):
        return (a > b).astype(np.int8)
    if isinstance(b, np.ndarray):
        return (a > b).astype(np.int8)
    return int(a > b)

def mul(a, b):
    # TODO: secure multiplication (Fmul)
    return a * b

def recip(x, eps=1e-12):
    # TODO: secure reciprocal (Frecip)
    return 1.0 / (x + eps)

def argmax(values):
    # TODO: secure argmax (Fargmax)
    return int(np.argmax(values))