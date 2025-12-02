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
        hidden_states: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        return self.weight * self.compute_loss(labels, logits, hidden_states, **kwargs)

    @abstractmethod
    def compute_loss(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        """Compute the auxiliary loss.

        Args:
            labels (torch.Tensor): Ground truth labels.
            logits (torch.Tensor): Model predictions.
            hidden_states (Sequence[torch.Tensor]): Hidden states from the model.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            float: Computed auxiliary loss.
        """
        ...
