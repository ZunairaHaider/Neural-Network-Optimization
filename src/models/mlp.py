"""Uniform-width fully connected MLP for regression.

Architecture: input(input_dim) → [Linear → Act] * n_layers → Linear → output(output_dim)
The output layer has NO activation — this is a regression model.

``input_dim`` and ``output_dim`` default to 1 for backward compatibility with
the original 1-D study. The 2-D and 3-D scaling-law sweeps set ``input_dim``
to the data dimension.

The flat-vector accessors (get_weight_vector / set_weights_from_vector) are
required by OBD and OBS, which work on the parameter space as a single
concatenated vector. The iteration order of parameters() is:
  layers[0].weight, layers[0].bias, layers[1].weight, layers[1].bias, ...
so the flat-vector layout is fully determined by the ModuleList order.

Adapted from pruning_study/models/mlp.py (flat-vector accessors kept verbatim;
sigmoid classification head replaced with linear regression head; fixed
activation replaced with configurable ReLU/Tanh/GELU).
"""

from __future__ import annotations

import torch
import torch.nn as nn

SUPPORTED_ACTIVATIONS = ("relu", "tanh", "gelu", "sigmoid")


class MLP(nn.Module):
    """Intentionally over-parameterised MLP for pruning experiments.

    Args:
        n_layers:   Number of hidden layers (each has width n_neurons).
        n_neurons:  Width of every hidden layer.
        activation: One of 'relu', 'tanh', 'gelu'.
    """

    _ACT_CLASSES = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}

    def __init__(
        self,
        n_layers: int = 3,
        n_neurons: int = 128,
        activation: str = "relu",
        input_dim: int = 1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        act_key = activation.lower()
        if act_key not in self._ACT_CLASSES:
            raise ValueError(
                f"activation must be one of {SUPPORTED_ACTIVATIONS}, got {activation!r}"
            )

        self._act = self._ACT_CLASSES[act_key]()

        # ModuleList so that OBD/OBS can index layers[i].weight / .bias directly.
        sizes = [input_dim] + [n_neurons] * n_layers + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

        self.config: dict = {
            "n_layers": n_layers,
            "n_neurons": n_neurons,
            "activation": activation,
            "input_dim": input_dim,
            "output_dim": output_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        return self.layers[-1](x)  # no activation — regression output

    # ------------------------------------------------------------------
    # Flat-vector accessors (required by OBD and OBS)
    # ------------------------------------------------------------------

    def get_weight_count(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_weight_vector(self) -> torch.Tensor:
        """Flatten all trainable parameters into one 1-D tensor (detached)."""
        with torch.no_grad():
            return torch.cat(
                [p.detach().reshape(-1) for p in self.parameters() if p.requires_grad]
            )

    def set_weights_from_vector(self, w: torch.Tensor) -> None:
        """Write a flat weight vector back into model parameters in-place.

        Args:
            w: 1-D tensor matching get_weight_count() in length.
        """
        expected = self.get_weight_count()
        if w.numel() != expected:
            raise ValueError(
                f"Weight vector has {w.numel()} entries but model has {expected}."
            )
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                if not p.requires_grad:
                    continue
                n = p.numel()
                p.copy_(w[offset : offset + n].view_as(p).to(p.device, p.dtype))
                offset += n
