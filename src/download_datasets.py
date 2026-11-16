"""Downloads the datasets used in the experiments into the data/ folder.

Usage:
    python src/download_datasets.py
"""

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DATASETS = {
    "iris.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "boston.csv": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    "mnist.npz": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
}


def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    for file_name in DATASETS:
        url = DATASETS[file_name]
        file_path = os.path.join(DATA_DIR, file_name)

        if os.path.exists(file_path):
            print("Already downloaded:", file_name)
            continue

        print("Downloading", file_name, "from", url)
        urllib.request.urlretrieve(url, file_path)

    print("Done. Files saved in", os.path.abspath(DATA_DIR))


if __name__ == "__main__":
    download_all()
