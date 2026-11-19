"""Multilayer Perceptron built on top of the autodiff engine."""

import json
import os

import numpy as np

from .autodiff import Tensor


def apply_activation(x, name):
    if name == "sigmoid":
        return x.sigmoid()
    if name == "tanh":
        return x.tanh()
    if name == "relu":
        return x.relu()
    if name == "leaky_relu":
        return x.leaky_relu()
    if name == "linear":
        return x
    raise ValueError("Unknown activation: " + name)


def init_weights(inputs, outputs, method, scale, rng):
    """Creates the weight matrix of a layer, with shape (inputs, outputs)."""
    if method == "he":
        return rng.normal(0.0, np.sqrt(2.0 / inputs), (inputs, outputs))
    if method == "xavier":
        limit = np.sqrt(6.0 / (inputs + outputs))
        return rng.uniform(-limit, limit, (inputs, outputs))
    if method == "normal":
        return rng.normal(0.0, scale, (inputs, outputs))
    if method == "uniform":
        return rng.uniform(-scale, scale, (inputs, outputs))
    raise ValueError("Unknown initialization: " + method)


class Layer:
    """Fully connected layer: activation(x @ W + b)."""

    def __init__(self, inputs, outputs, activation, init, init_scale, rng):
        self.W = Tensor(init_weights(inputs, outputs, init, init_scale, rng))
        self.b = Tensor(np.zeros((1, outputs)))
        self.activation = activation

    def forward(self, x):
        return apply_activation(x @ self.W + self.b, self.activation)


class MLP:
    """Multilayer Perceptron.

    Example:
        model = MLP(input_size=4, hidden_sizes=[16, 16], output_size=3,
                    hidden_activation="relu", output_activation="sigmoid", init="he")

    hidden_sizes:       units in each hidden layer (its length is the number of hidden layers)
    hidden_activation:  "sigmoid", "tanh", "relu", "leaky_relu" or "linear"
    output_activation:  same options as above
    init:               "he", "xavier", "normal" or "uniform"
    init_scale:         standard deviation of "normal" / limit of "uniform"
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 hidden_activation="relu", output_activation="linear",
                 init="he", init_scale=0.1, seed=0):
        self.config = {
            "input_size": input_size,
            "hidden_sizes": list(hidden_sizes),
            "output_size": output_size,
            "hidden_activation": hidden_activation,
            "output_activation": output_activation,
            "init": init,
            "init_scale": init_scale,
            "seed": seed,
        }

        rng = np.random.default_rng(seed)
        self.layers = []

        inputs = input_size
        for units in hidden_sizes:
            self.layers.append(Layer(inputs, units, hidden_activation, init, init_scale, rng))
            inputs = units
        self.layers.append(Layer(inputs, output_size, output_activation, init, init_scale, rng))

        # Outputs of every layer in the last forward pass (used for monitoring).
        self.activations = []

    def forward(self, x):
        """Forward pass. Returns a Tensor connected to the computational graph."""
        if not isinstance(x, Tensor):
            x = Tensor(x)

        self.activations = []
        for layer in self.layers:
            x = layer.forward(x)
            self.activations.append(x)
        return x

    def predict(self, x):
        """Inference. Returns a Numpy array."""
        return self.forward(x).data

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params

    # -- Export / import

    def save(self, path):
        """Saves topology and weights in a single .npz file.

        The file contains:
          - "config": JSON text with the topology (sizes, activations, ...);
          - "W0", "b0", "W1", "b1", ...: weights and biases of each layer.
        """
        folder = os.path.dirname(path)
        if folder != "":
            os.makedirs(folder, exist_ok=True)

        arrays = {"config": json.dumps(self.config)}
        for i in range(len(self.layers)):
            arrays["W" + str(i)] = self.layers[i].W.data
            arrays["b" + str(i)] = self.layers[i].b.data
        np.savez(path, **arrays)

    @staticmethod
    def load(path):
        """Rebuilds a trained model from a file created by save()."""
        file = np.load(path)
        config = json.loads(str(file["config"]))

        model = MLP(**config)
        for i in range(len(model.layers)):
            model.layers[i].W.data = file["W" + str(i)]
            model.layers[i].b.data = file["b" + str(i)]
        return model
