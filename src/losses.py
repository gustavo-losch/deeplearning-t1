"""Cost functions, written only with operations of the autodiff engine."""

from .autodiff import to_tensor


def mse(prediction, target):
    """Mean Squared Error."""
    target = to_tensor(target)
    difference = prediction - target
    return (difference ** 2).mean()


def bce(prediction, target):
    """Binary Cross-Entropy. Predictions must be between 0 and 1 (sigmoid output)."""
    target = to_tensor(target)
    eps = 1e-9  # avoids log(0)
    loss = target * (prediction + eps).log() + (1.0 - target) * (1.0 - prediction + eps).log()
    return -loss.mean()


def compute_loss(name, prediction, target):
    if name == "mse":
        return mse(prediction, target)
    if name == "bce":
        return bce(prediction, target)
    raise ValueError("Unknown loss: " + name)
