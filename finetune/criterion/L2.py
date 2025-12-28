import torch

from .base import Criterion


class L2HiddenStateNorm(Criterion):
    """Auxiliary loss that penalizes the L2 norm of the last hidden state."""

    def __init__(self, weight: float = 1.0, reduce: str = "mean") -> None:
        super().__init__(weight=weight)
        if reduce not in {"mean", "sum"}:
            msg = 'reduce must be either "mean" or "sum"'
            raise ValueError(msg)
        self.reduce = reduce

    def compute_loss(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        if last_hidden_state is None:
            return torch.tensor(0.0, device="cpu")

        l2 = last_hidden_state.float().pow(2)
        if self.reduce == "mean":
            return l2.mean()
        return l2.sum()
