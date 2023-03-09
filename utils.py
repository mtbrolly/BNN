import numpy as np


def generate_synthetic_data_A(N, seed=1):
    rng = np.random.default_rng(seed=seed)
    X = rng.normal(size=N)[:, None]
    Y = 0.1 * rng.normal(size=N)[:, None] + np.sin(X)
    return X, Y


def generate_synthetic_data_B(N, seed=1):
    rng = np.random.default_rng(seed=seed)
    X = rng.normal(size=N)[:, None] * 3
    Y = rng.normal(size=N)[:, None] * (1 + 0.5 * np.sin(X))
    return X, Y


def generate_synthetic_data_C(N, seed=1):
    rng = np.random.default_rng(seed=seed)
    # X = rng.normal(size=N)[:, None]
    X = rng.uniform(size=N)[:, None]
    Y = rng.normal(size=N)[:, None] * (1 + X ** 2.)
    return X, Y
