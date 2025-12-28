import torch

from .base import Criterion


class L1Norm(Criterion):
    """Auxiliary loss that penalizes the L1 norm of the last hidden state."""

    def compute_loss(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        l1 = last_hidden_state.float().abs()
        return self.reduction(l1)


class L2Norm(Criterion):
    """Auxiliary loss that penalizes the L2 norm of the last hidden state."""

    def compute_loss(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        l2 = last_hidden_state.float().pow(2)
        return self.reduction(l2)
