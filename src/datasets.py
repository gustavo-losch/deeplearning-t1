"""Loads the datasets from the data/ folder (run download_datasets.py first)."""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def one_hot(labels, n_classes):
    encoded = np.zeros((len(labels), n_classes))
    for i in range(len(labels)):
        encoded[i, int(labels[i])] = 1.0
    return encoded


def load_iris():
    """Returns X (150, 4) and y (150, 3) one-hot encoded."""
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    features = []
    labels = []

    with open(os.path.join(DATA_DIR, "iris.csv")) as file:
        for line in file:
            line = line.strip()
            if line == "":
                continue
            values = line.split(",")
            features.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
            labels.append(class_names.index(values[4]))

    return np.array(features), one_hot(labels, 3)


def load_boston():
    """Returns X (506, 13) and y (506, 1), the median house value."""
    rows = []

    with open(os.path.join(DATA_DIR, "boston.csv")) as file:
        file.readline()  # skip the header
        for line in file:
            line = line.strip().replace('"', "")
            if line == "":
                continue
            row = []
            for value in line.split(","):
                row.append(float(value))
            rows.append(row)

    data = np.array(rows)
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


def load_mnist():
    """Returns X_train (60000, 784), y_train (60000, 10), X_test (10000, 784), y_test (10000, 10).

    Pixels are scaled to [0, 1] and labels are one-hot encoded.
    """
    file = np.load(os.path.join(DATA_DIR, "mnist.npz"))
    X_train = file["x_train"].reshape(-1, 784) / 255.0
    X_test = file["x_test"].reshape(-1, 784) / 255.0
    y_train = one_hot(file["y_train"], 10)
    y_test = one_hot(file["y_test"], 10)
    return X_train, y_train, X_test, y_test


def train_val_split(X, y, val_fraction=0.2, seed=0):
    """Shuffles the data and splits it into training and validation sets."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    n_val = int(X.shape[0] * val_fraction)

    val_index = order[:n_val]
    train_index = order[n_val:]
    return X[train_index], y[train_index], X[val_index], y[val_index]


def standardize(train, val):
    """Scales both sets to zero mean and unit variance, using training statistics."""
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (val - mean) / std
