from collections.abc import Sequence
from typing import Any

import torch

from .base import Criterion


class EmptyCriterion(Criterion):
    """A criterion that returns zero loss."""

    def compute_loss(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        last_hidden_state: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        return 0.0
