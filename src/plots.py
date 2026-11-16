"""Monitoring plots. Every function saves the figure when save_path is given."""

import os

import matplotlib.pyplot as plt
import numpy as np


def finish_figure(figure, save_path):
    figure.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        figure.savefig(save_path, dpi=120)
    plt.show()


def plot_curves(train_values, val_values, y_label, title, save_path=None):
    epochs = range(1, len(train_values) + 1)

    figure = plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_values, label="train")
    plt.plot(epochs, val_values, label="validation")
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    finish_figure(figure, save_path)


def plot_loss(history, title="Loss", save_path=None):
    """Cost function vs. epoch, for training and validation."""
    plot_curves(history["train_loss"], history["val_loss"], "loss", title, save_path)


def plot_metric(history, metric_name, title=None, save_path=None):
    """Evaluation metric vs. epoch, for training and validation."""
    if title is None:
        title = metric_name
    plot_curves(history["train_metric"], history["val_metric"], metric_name, title, save_path)


def plot_histograms(history, kind, title=None, save_path=None):
    """Histograms of each layer along training.

    kind: "activations" or "gradients" (gradients of the loss w.r.t. the weights).
    Each row is a moment of the training, each column is a layer.
    """
    snapshots = history["snapshots"]
    n_rows = len(snapshots)
    n_cols = len(snapshots[0][kind])

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), squeeze=False)

    for row in range(n_rows):
        snapshot = snapshots[row]
        for col in range(n_cols):
            ax = axes[row][col]
            ax.hist(snapshot[kind][col], bins=40)
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
            ax.xaxis.get_offset_text().set_fontsize(7)
            if row == 0:
                ax.set_title("layer " + str(col + 1), fontsize=9)
            if col == 0:
                ax.set_ylabel("epoch " + str(snapshot["epoch"]), fontsize=9)

    if title is None:
        title = "Histogram of " + kind
    figure.suptitle(title)
    finish_figure(figure, save_path)


def plot_gradient_sizes(history, title="Gradient size per layer", save_path=None):
    """Mean absolute gradient of each layer (log scale), one line per snapshot."""
    figure = plt.figure(figsize=(6, 4))

    for snapshot in history["snapshots"]:
        sizes = []
        for gradient in snapshot["gradients"]:
            sizes.append(np.mean(np.abs(gradient)))
        layers = range(1, len(sizes) + 1)
        plt.plot(layers, sizes, marker="o", label="epoch " + str(snapshot["epoch"]))

    plt.yscale("log")
    plt.xlabel("layer (1 = closest to the input)")
    plt.ylabel("mean |gradient|")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    finish_figure(figure, save_path)
