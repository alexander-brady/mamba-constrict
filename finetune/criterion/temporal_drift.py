import torch

from .base import Criterion


class TemporalDrift(Criterion):
    """Penalize rapid changes across the sequence in the last hidden state."""

    def __init__(self, weight: float = 1.0, reduce: str = "mean", order: int = 1) -> None:
        super().__init__(weight=weight)
        if reduce not in {"mean", "sum"}:
            msg = 'reduce must be either "mean" or "sum"'
            raise ValueError(msg)
        if order < 1:
            msg = "order must be >= 1"
            raise ValueError(msg)
        self.reduce = reduce
        self.order = order

    def compute_loss(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        if last_hidden_state is None:
            return torch.tensor(0.0, device="cpu")

        diff = last_hidden_state.float()
        for _ in range(self.order):
            # difference along sequence dimension
            diff = diff[:, 1:, :] - diff[:, :-1, :]

        penalty = diff.pow(2)
        if self.reduce == "mean":
            return penalty.mean()
        return penalty.sum()

