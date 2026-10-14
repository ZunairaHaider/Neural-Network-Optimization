# ============================================================================
# dataset_loader.py
# ============================================================================

import torch


def load_pt_dataset(path):

    data = torch.load(path)

    if isinstance(data, dict):

        possible_x = ["X", "x", "inputs", "features"]
        possible_y = ["Y", "y", "targets", "labels"]

        X, Y = None, None

        for k in possible_x:
            if k in data:
                X = data[k]; break

        for k in possible_y:
            if k in data:
                Y = data[k]; break

    elif isinstance(data, (tuple, list)):
        X, Y = data

    else:
        raise ValueError("Unsupported dataset format")

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    if X.shape[0] > X.shape[1]:
        X = X.T

    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    return X, Y
