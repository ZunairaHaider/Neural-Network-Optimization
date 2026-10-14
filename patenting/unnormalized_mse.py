import argparse
import os
import numpy as np

import sys

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from patenting.dataset_loader import load_pt_dataset
from patenting.utils import normalize_data

DATASETS = {
    "dataset_A": {
        "train": "CS_ANN/data/processed/dataset_A_train.pt",
        "test":  "CS_ANN/data/processed/dataset_A_test.pt",
    },
    "dataset_B": {
        "train": "CS_ANN/data/processed/dataset_B_train.pt",
        "test":  "CS_ANN/data/processed/dataset_B_test.pt",
    },
    "dataset_A_2d": {
        "train": "CS_ANN/data/processed/dataset_A_2d_train.pt",
        "test":  "CS_ANN/data/processed/dataset_A_2d_test.pt",
    },
    "dataset_B_2d": {
        "train": "CS_ANN/data/processed/dataset_B_2d_train.pt",
        "test":  "CS_ANN/data/processed/dataset_B_2d_test.pt",
    },
    "dataset_A_3d": {
        "train": "CS_ANN/data/processed/dataset_A_3d_train.pt",
        "test":  "CS_ANN/data/processed/dataset_A_3d_test.pt",
    },
    "dataset_B_3d": {
        "train": "CS_ANN/data/processed/dataset_B_3d_train.pt",
        "test":  "CS_ANN/data/processed/dataset_B_3d_test.pt",
    },
}


def compute_unnormalized_mse(normalized_mse, std_y):
    """Convert normalized MSE to unnormalized MSE using output scaling."""
    std_y = np.asarray(std_y).reshape(-1)
    if std_y.size == 1:
        return float(normalized_mse * std_y[0] ** 2)
    return float(normalized_mse * np.mean(std_y ** 2))


def load_output_std(dataset_name, train_path):
    _, Y_train = load_pt_dataset(train_path)
    _, _Y_test, std_y = normalize_data(Y_train, Y_train)
    return std_y.squeeze()


def main():
    parser = argparse.ArgumentParser(
        description="Convert normalized MSE into unnormalized MSE for CS_ANN datasets."
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        default=None,
        help="Dataset name from the main demo (dataset_A, dataset_B, dataset_A_2d, ...).",
    )
    parser.add_argument(
        "--normalized_mse",
        type=float,
        nargs="?",
        default=None,
        help="Normalized MSE value to convert to unnormalized MSE.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute unnormalized MSE for all built-in datasets using the same normalized MSE value.",
    )

    args = parser.parse_args()

    if args.normalized_mse is None:
        parser.error("Please provide a normalized MSE value.")

    if args.all:
        dataset_items = DATASETS.items()
    else:
        if args.dataset is None:
            parser.error("Please provide a dataset name or use --all.")
        if args.dataset not in DATASETS:
            parser.error(f"Unknown dataset '{args.dataset}'. Available: {', '.join(DATASETS)}")
        dataset_items = [(args.dataset, DATASETS[args.dataset])]

    for dataset_name, paths in dataset_items:
        train_path = paths["train"]
        if not os.path.exists(train_path):
            print(f"Skipping {dataset_name}: train file not found at {train_path}")
            continue

        try:
            std_y = load_output_std(dataset_name, train_path)
        except Exception as exc:
            print(f"Failed to load {dataset_name}: {exc}")
            continue

        unnormalized_mse = compute_unnormalized_mse(args.normalized_mse, std_y)

        print(f"{dataset_name}")
        print(f"  output std shape: {std_y.shape}")
        print(f"  output std: {std_y}")
        print(f"  normalized MSE: {args.normalized_mse:.6e}")
        print(f"  unnormalized MSE: {unnormalized_mse:.6e}\n")


if __name__ == "__main__":
    main()
