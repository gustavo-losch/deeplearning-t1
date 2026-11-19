"""Automatic differentiation engine built on Numpy.

Every operation between Tensors creates a new Tensor that remembers:
  - its parents (the Tensors used to compute it);
  - a small function (backward_step) that knows how to send its gradient
    back to the parents.

This forms a computational graph. Calling .backward() on the final Tensor
(usually the loss) walks the graph in reverse order and fills the .grad of
every Tensor that took part in the computation.
"""

import numpy as np


def to_tensor(value):
    """Wraps numbers and Numpy arrays into a Tensor (Tensors pass through)."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


def unbroadcast(grad, shape):
    """Undoes Numpy broadcasting so the gradient has the given shape.

    When a small array (or a scalar) is broadcast to a bigger shape, each of
    its values is used many times, so its gradient is the sum of the
    gradients of all the copies.
    """
    # Remove the extra leading dimensions created by broadcasting.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum over the dimensions that had size 1 and were stretched.
    for axis in range(len(shape)):
        if shape[axis] == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Tensor:
    # Makes "numpy_array + tensor" call Tensor.__radd__ instead of Numpy's own code.
    __array_ufunc__ = None

    def __init__(self, data, parents=(), op=""):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self.op = op
        self.backward_step = None

    def __repr__(self):
        return "Tensor(shape=" + str(self.data.shape) + ", op='" + self.op + "')"

    @property
    def shape(self):
        return self.data.shape

    # -- Backward pass

    def backward(self):
        """Computes the gradient of this Tensor with respect to every Tensor in the graph."""
        # Topological order: every Tensor appears after all of its parents.
        order = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            order.append(node)

        visit(self)

        # d(self)/d(self) = 1
        self.grad = np.ones_like(self.data)

        # Walk the graph from the output back to the inputs.
        for node in reversed(order):
            if node.backward_step is not None:
                node.backward_step()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # -- Addition, subtraction, product and division (with broadcasting)

    def __add__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def backward_step():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out.backward_step = backward_step
        return out

    def __sub__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data - other.data, (self, other), "-")

        def backward_step():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(-out.grad, other.data.shape)

        out.backward_step = backward_step
        return out

    def __mul__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def backward_step():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out.backward_step = backward_step
        return out

    def __truediv__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def backward_step():
            self.grad += unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out.backward_step = backward_step
        return out

    def __neg__(self):
        return self * -1.0

    # Versions used when the Tensor is on the right side, e.g. "2 + tensor".
    def __radd__(self, other):
        return to_tensor(other) + self

    def __rsub__(self, other):
        return to_tensor(other) - self

    def __rmul__(self, other):
        return to_tensor(other) * self

    def __rtruediv__(self, other):
        return to_tensor(other) / self

    # -- Matrix multiplication

    def __matmul__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def backward_step():
            a, b, g = self.data, other.data, out.grad

            # Dot product: both operands are vectors.
            if a.ndim == 1 and b.ndim == 1:
                self.grad += g * b
                other.grad += g * a
                return

            # Promote 1D operands to 2D so the same formulas work for every case.
            a2 = a[None, :] if a.ndim == 1 else a
            b2 = b[:, None] if b.ndim == 1 else b
            g2 = g
            if a.ndim == 1:
                g2 = np.expand_dims(g2, -2)
            if b.ndim == 1:
                g2 = np.expand_dims(g2, -1)

            grad_a = unbroadcast(g2 @ np.swapaxes(b2, -1, -2), a2.shape)
            grad_b = unbroadcast(np.swapaxes(a2, -1, -2) @ g2, b2.shape)

            self.grad += grad_a.reshape(a.shape)
            other.grad += grad_b.reshape(b.shape)

        out.backward_step = backward_step
        return out

    def __rmatmul__(self, other):
        return to_tensor(other) @ self

    # -- Powers, roots, exponentials and logarithms

    def __pow__(self, exponent):
        # tensor ** tensor is written with other operations of the engine:
        # a ** b = exp(b * ln(a))
        if isinstance(exponent, Tensor):
            return (exponent * self.log()).exp()

        # tensor ** number (the exponent is a constant)
        out = Tensor(self.data ** exponent, (self,), "**")

        def backward_step():
            self.grad += out.grad * exponent * self.data ** (exponent - 1)

        out.backward_step = backward_step
        return out

    def __rpow__(self, base):
        # number ** tensor = exp(tensor * ln(number))
        return (self * np.log(base)).exp()

    def sqrt(self):
        return self ** 0.5

    def root(self, n):
        """n-th root."""
        return self ** (1.0 / n)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def backward_step():
            self.grad += out.grad * out.data

        out.backward_step = backward_step
        return out

    def log(self, base=None):
        """Natural logarithm, or logarithm in the given base."""
        if base is not None:
            return self.log() / np.log(base)

        out = Tensor(np.log(self.data), (self,), "log")

        def backward_step():
            self.grad += out.grad / self.data

        out.backward_step = backward_step
        return out

    # -- Summation

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def backward_step():
            grad = out.grad
            # Put back the dimension removed by the sum, so it can be broadcast.
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out.backward_step = backward_step
        return out

    def mean(self):
        return self.sum() / self.data.size

    # -- Activation functions

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def backward_step():
            self.grad += out.grad * (self.data > 0)

        out.backward_step = backward_step
        return out

    def leaky_relu(self, slope=0.01):
        out = Tensor(np.where(self.data > 0, self.data, slope * self.data), (self,), "leaky_relu")

        def backward_step():
            self.grad += out.grad * np.where(self.data > 0, 1.0, slope)

        out.backward_step = backward_step
        return out

    def sigmoid(self):
        # The clip avoids overflow in exp for very negative inputs.
        clipped = np.clip(self.data, -500, 500)
        out = Tensor(1.0 / (1.0 + np.exp(-clipped)), (self,), "sigmoid")

        def backward_step():
            self.grad += out.grad * out.data * (1.0 - out.data)

        out.backward_step = backward_step
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), "tanh")

        def backward_step():
            self.grad += out.grad * (1.0 - out.data ** 2)

        out.backward_step = backward_step
        return out
