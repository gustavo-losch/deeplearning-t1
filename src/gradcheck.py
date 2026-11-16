"""Checks the gradients of the autodiff engine against finite differences."""

import numpy as np


def numerical_gradient(function, tensor, eps=1e-6):
    """Approximates d(function)/d(tensor) with central finite differences.

    `function` takes no arguments and returns a scalar Tensor.
    """
    # Makes scalars behave like arrays, so they can be changed in place.
    tensor.data = np.atleast_1d(tensor.data)

    gradient = np.zeros_like(tensor.data)
    flat_data = tensor.data.reshape(-1)
    flat_gradient = gradient.reshape(-1)

    for i in range(flat_data.size):
        original_value = flat_data[i]

        flat_data[i] = original_value + eps
        value_plus = float(function().data.sum())

        flat_data[i] = original_value - eps
        value_minus = float(function().data.sum())

        flat_data[i] = original_value
        flat_gradient[i] = (value_plus - value_minus) / (2 * eps)

    return gradient


def check_gradient(function, tensors, eps=1e-6):
    """Returns the largest difference between autodiff and numerical gradients.

    `tensors` is the list of Tensors whose gradients will be checked.
    """
    for tensor in tensors:
        tensor.data = np.atleast_1d(tensor.data)
        tensor.zero_grad()

    # Gradients from the engine.
    output = function()
    output.backward()

    largest_error = 0.0
    for tensor in tensors:
        expected = numerical_gradient(function, tensor, eps)
        error = float(np.max(np.abs(expected - tensor.grad)))
        if error > largest_error:
            largest_error = error

    return largest_error
