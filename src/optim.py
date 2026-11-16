"""Optimizers: they update the parameters using the gradients from the engine."""

import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        for param in self.parameters:
            param.data = param.data - self.lr * param.grad


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Moving averages of the gradient (m) and of the squared gradient (v).
        self.m = []
        self.v = []
        for param in parameters:
            self.m.append(np.zeros_like(param.data))
            self.v.append(np.zeros_like(param.data))

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        self.t += 1

        for i in range(len(self.parameters)):
            param = self.parameters[i]

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad ** 2

            # Bias correction (m and v start at zero).
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
