from typing import Literal

import torch

from .base import Criterion


class TemporalDrift(Criterion):
    """Penalize rapid changes across the sequence in the last hidden state."""

    def __init__(
        self,
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        order: int = 1,
    ) -> None:
        super().__init__(weight=weight, reduction=reduction)
        if order < 1:
            raise ValueError("Order must be >= 1")
        self.order = order

    def compute_loss(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        diff = last_hidden_state.float()
        for _ in range(self.order):
            # difference along sequence dimension
            diff = diff[:, 1:, :] - diff[:, :-1, :]

        penalty = diff.pow(2)
        return self.reduction(penalty)
