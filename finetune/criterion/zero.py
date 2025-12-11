import torch

from .base import Criterion


class EmptyCriterion(Criterion):
    """A criterion that returns zero loss."""

    def compute_loss(self, last_hidden_state: torch.Tensor) -> float:
        return 0.0
