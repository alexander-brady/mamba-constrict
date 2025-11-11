from collections.abc import Sequence
from typing import Any

import torch

from .base import Criterion


class EmptyCriterion(Criterion):
    def compute_loss(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        return 0.0
