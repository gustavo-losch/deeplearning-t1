# deeplearning-t1

MLP with automatic differentiation, written only with Numpy (see `assets/statement.pdf`).

## Structure

| Path | What it has |
| --- | --- |
| `src/autodiff.py` | `Tensor`: the autodiff engine (computational graph + backward pass) |
| `src/gradcheck.py` | Gradient check with finite differences |
| `src/nn.py` | `MLP` (activations, initializations, save / load) |
| `src/losses.py` | BCE and MSE |
| `src/optim.py` | SGD and Adam |
| `src/training.py` | Training loop and monitoring history |
| `src/plots.py` | Monitoring plots (loss, metric, histograms) |
| `src/datasets.py` | Loaders for Iris, Boston Housing and MNIST |
| `src/download_datasets.py` | Downloads the datasets into `data/` |
| `notebooks/` | Experiments (already executed) |
| `assets/plots/` | Plots saved by the notebooks |
| `models/` | Trained models exported by the notebooks |

## Running

```bash
pip install -r requirements.txt
python src/download_datasets.py
jupyter notebook notebooks/
```

## Example

```python
from src.nn import MLP
from src.optim import Adam
from src.training import train

# X_train / X_val are already preprocessed (see the notebooks).
model = MLP(input_size=4, hidden_sizes=[8], output_size=3,
            hidden_activation="tanh", output_activation="sigmoid", init="xavier")
optimizer = Adam(model.parameters(), lr=0.01)
history = train(model, optimizer, "bce", X_train, y_train, X_val, y_val, epochs=100)

model.save("models/my_model.npz")
model = MLP.load("models/my_model.npz")
predictions = model.predict(X_val)
```

## Preprocessing

`MLP` stores only the topology and weights, as required by the assignment. The data preprocessing is
not embedded in the model file, and `predict` expects inputs that were transformed in the same way as
during training.

The notebooks standardize the features with the training statistics:

```python
X_train, X_val = standardize(X_train, X_val)
predictions = model.predict(X_val)
```

For inference from raw values, apply the training mean and standard deviation first:

```python
# raw_X_train is the training set before standardization.
feature_mean = raw_X_train.mean(axis=0)
feature_std = raw_X_train.std(axis=0) + 1e-8
scaled_X = (raw_X - feature_mean) / feature_std
predictions = model.predict(scaled_X)
```

In the Boston example the target is standardized as well, so the output must be converted back to the
original scale. Keep the preprocessing statistics in a separate file or pipeline if a self-contained
inference artifact is needed; they do not belong to the network itself.
