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

model = MLP(input_size=4, hidden_sizes=[8], output_size=3,
            hidden_activation="tanh", output_activation="sigmoid", init="xavier")
optimizer = Adam(model.parameters(), lr=0.01)
history = train(model, optimizer, "bce", X_train, y_train, X_val, y_val, epochs=100)

model.save("models/my_model.npz")
model = MLP.load("models/my_model.npz")
predictions = model.predict(X_val)
```
