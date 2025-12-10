from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

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

    def __call__(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        last_hidden_state: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        return self.weight * self.compute_loss(labels, logits, last_hidden_state, **kwargs)

    @abstractmethod
    def compute_loss(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        last_hidden_state: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        """Compute the auxiliary loss.

        Args:
            labels (torch.Tensor): Ground truth labels.
            logits (torch.Tensor): Model predictions.
            last_hidden_state (torch.Tensor): Last hidden state from the model [B, T, D].
            **kwargs (Any): Additional keyword arguments.

        Returns:
            float: Computed auxiliary loss.
        """
        ...
