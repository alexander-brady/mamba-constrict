from abc import ABC, abstractmethod

import torch


class Criterion(ABC):
    """Abstract base class for auxiliary loss functions.

    Args:
        weight (float): Weighting factor for the loss. Default is 1.0.

    Methods:
        compute_loss(labels, preds, hidden_states, **kwargs) -> float:
            Abstract method to compute the auxiliary loss. Must be implemented by subclasses.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def __call__(self, last_hidden_state: torch.Tensor) -> float:
        return self.weight * self.compute_loss(last_hidden_state)

    @abstractmethod
    def compute_loss(self, last_hidden_state: torch.Tensor) -> float:
        """Compute the auxiliary loss.

        Args:
            last_hidden_state (torch.Tensor): Last hidden state from the model [B, T, D].
            **kwargs (Any): Additional keyword arguments.

        Returns:
            float: Computed auxiliary loss.
        """
        ...
