"""Training loop with monitoring (loss, metric, activations and gradients)."""

import numpy as np

from .losses import compute_loss


def accuracy(prediction, target):
    if target.shape[1] == 1:
        # Binary problem: one output, threshold at 0.5.
        predicted_class = prediction > 0.5
        true_class = target > 0.5
    else:
        # Multi-class problem: one-hot targets, the largest output wins.
        predicted_class = np.argmax(prediction, axis=1)
        true_class = np.argmax(target, axis=1)
    return float(np.mean(predicted_class == true_class))


def r2_score(prediction, target):
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - np.mean(target, axis=0)) ** 2)
    return float(1.0 - residual / total)


def compute_metric(name, prediction, target):
    if name == "accuracy":
        return accuracy(prediction, target)
    if name == "r2":
        return r2_score(prediction, target)
    raise ValueError("Unknown metric: " + name)


def evaluate(model, X, y, loss_name, metric_name):
    """Returns (loss, metric) of the model on the given data."""
    prediction = model.forward(X)
    loss = float(compute_loss(loss_name, prediction, y).data)
    metric = compute_metric(metric_name, prediction.data, y)
    return loss, metric


def take_snapshot(model, optimizer, X, y, loss_name, epoch):
    """Stores the activations and the weight gradients of every layer."""
    optimizer.zero_grad()
    prediction = model.forward(X)
    loss = compute_loss(loss_name, prediction, y)
    loss.backward()

    activations = []
    gradients = []
    for i in range(len(model.layers)):
        activations.append(model.activations[i].data.flatten())
        gradients.append(model.layers[i].W.grad.flatten())

    optimizer.zero_grad()
    return {"epoch": epoch, "activations": activations, "gradients": gradients}


def train(model, optimizer, loss_name, X_train, y_train, X_val, y_val,
          epochs=100, batch_size=32, metric_name="accuracy", seed=0, print_every=10):
    """Trains the model and returns the history used by the monitoring plots.

    loss_name:    "bce" or "mse"
    metric_name:  "accuracy" or "r2"
    """
    rng = np.random.default_rng(seed)
    n_samples = X_train.shape[0]

    history = {
        "train_loss": [], "val_loss": [],
        "train_metric": [], "val_metric": [],
        "snapshots": [],
    }

    # Activations and gradients are stored at the start, middle and end of training.
    snapshot_epochs = [1, max(1, epochs // 2), epochs]
    snapshot_size = min(512, n_samples)

    for epoch in range(1, epochs + 1):
        if epoch in snapshot_epochs:
            snapshot = take_snapshot(model, optimizer, X_train[:snapshot_size],
                                     y_train[:snapshot_size], loss_name, epoch)
            history["snapshots"].append(snapshot)

        # One pass over the training data, in shuffled mini-batches.
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = order[start:start + batch_size]

            optimizer.zero_grad()
            prediction = model.forward(X_train[batch])
            loss = compute_loss(loss_name, prediction, y_train[batch])
            loss.backward()
            optimizer.step()

        # Monitoring at the end of the epoch.
        train_loss, train_metric = evaluate(model, X_train, y_train, loss_name, metric_name)
        val_loss, val_metric = evaluate(model, X_val, y_val, loss_name, metric_name)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print("epoch %4d | train loss %.4f | val loss %.4f | train %s %.4f | val %s %.4f"
                  % (epoch, train_loss, val_loss, metric_name, train_metric, metric_name, val_metric))

    return history
