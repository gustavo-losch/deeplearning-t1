"""Monitoring plots. Every function saves the figure when save_path is given."""

import os

import matplotlib.pyplot as plt
import numpy as np


def finish_figure(figure, save_path):
    figure.tight_layout()
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        figure.savefig(save_path, dpi=120)
    plt.show()


def plot_curves(train_values, val_values, y_label, title, save_path=None, ylim=None, log_y=False):
    epochs = range(1, len(train_values) + 1)

    figure = plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_values, label="train")
    plt.plot(epochs, val_values, label="validation")
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.title(title)
    if log_y:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(alpha=0.3)
    finish_figure(figure, save_path)


def plot_loss(history, title="Loss", save_path=None, ylim=None, log_y=False):
    """Cost function vs. epoch, for training and validation.

    ylim: (bottom, top) of the y axis. Without it matplotlib zooms into the data,
          so a nearly flat curve can look like it moves a lot.
    log_y: log scale on the y axis (useful when losses differ by orders of magnitude).
    """
    plot_curves(history["train_loss"], history["val_loss"], "loss", title, save_path, ylim, log_y)


def plot_metric(history, metric_name, title=None, save_path=None, ylim=None):
    """Evaluation metric vs. epoch, for training and validation."""
    if title is None:
        title = metric_name
    plot_curves(history["train_metric"], history["val_metric"], metric_name, title, save_path, ylim)


def plot_histograms(history, kind, title=None, save_path=None, share_x=None):
    """Histograms of each layer along training.

    kind: "activations" or "gradients" (gradients of the loss w.r.t. the weights).
    Each row is a moment of the training, each column is a layer.
    share_x: True:     every panel uses the same x range (and the same bins), so the
                       panels can be compared directly;
             "column": each layer keeps the same x range along the epochs;
             False:    every panel has its own range.
             Defaults to True for activations and False for gradients (their sizes
             differ by orders of magnitude).
    """
    if share_x is None:
        share_x = kind == "activations"

    snapshots = history["snapshots"]
    n_rows = len(snapshots)
    n_cols = len(snapshots[0][kind])

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), squeeze=False)

    def value_range(columns):
        lowest = min(float(np.min(snapshot[kind][col])) for snapshot in snapshots for col in columns)
        highest = max(float(np.max(snapshot[kind][col])) for snapshot in snapshots for col in columns)
        if lowest == highest:
            lowest, highest = lowest - 0.5, highest + 0.5
        return (lowest, highest)

    for row in range(n_rows):
        snapshot = snapshots[row]
        for col in range(n_cols):
            if share_x == "column":
                x_range = value_range([col])
            elif share_x:
                x_range = value_range(range(n_cols))
            else:
                x_range = None
            ax = axes[row][col]
            ax.hist(snapshot[kind][col], bins=40, range=x_range)
            if x_range is not None:
                ax.set_xlim(x_range)
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


def plot_gradient_sizes(history, title="Gradient size per layer", save_path=None, ylim=None):
    """Mean absolute gradient of each layer (log scale), one line per snapshot."""
    figure = plt.figure(figsize=(6, 4))

    for snapshot in history["snapshots"]:
        sizes = []
        for gradient in snapshot["gradients"]:
            sizes.append(np.mean(np.abs(gradient)))
        layers = range(1, len(sizes) + 1)
        plt.plot(layers, sizes, marker="o", label="epoch " + str(snapshot["epoch"]))

    plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("layer (1 = closest to the input)")
    plt.ylabel("mean |gradient|")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    finish_figure(figure, save_path)
