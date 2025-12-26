import torch

from .base import Criterion


class MahalanobisEnergy(Criterion):
    """Mahalanobis distance regularizer on the last hidden state.

    Uses diagonal covariance estimated via EMA. Penalizes deviations from the
    running mean normalized by running variance for each hidden dimension.
    """

    def __init__(
        self,
        weight: float = 1.0,
        reduce: str = "mean",
        alpha: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(weight=weight)
        if reduce not in {"mean", "sum"}:
            msg = 'reduce must be either "mean" or "sum"'
            raise ValueError(msg)
        if not 0 < alpha <= 1:
            msg = "alpha must be in (0, 1]"
            raise ValueError(msg)
        if eps <= 0:
            msg = "eps must be positive"
            raise ValueError(msg)

        self.reduce = reduce
        self.alpha = alpha
        self.eps = eps

        self._mean: torch.Tensor | None = None
        self._var: torch.Tensor | None = None
        self._initialized = False

    def _initialize_buffers(self, last_hidden_state: torch.Tensor) -> None:
        hidden_dim = last_hidden_state.shape[-1]
        device = last_hidden_state.device
        self._mean = torch.zeros(hidden_dim, device=device)
        self._var = torch.ones(hidden_dim, device=device)
        self._initialized = True

    def _update_buffers(self, batch_mean: torch.Tensor, batch_var: torch.Tensor) -> None:
        self._mean = self.alpha * self._mean + (1 - self.alpha) * batch_mean
        self._var = self.alpha * self._var + (1 - self.alpha) * batch_var

    def compute_loss(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        if last_hidden_state is None:
            return torch.tensor(0.0, device="cpu")

        hs = last_hidden_state.float()  # (B, S, D)
        if not self._initialized:
            self._initialize_buffers(hs)

        # Batch stats over batch and sequence
        batch_mean = hs.mean(dim=(0, 1))  # (D,)
        batch_var = hs.var(dim=(0, 1), unbiased=False)  # (D,)

        self._update_buffers(batch_mean, batch_var)

        mean = self._mean.view(1, 1, -1)
        var = self._var.view(1, 1, -1)
        mahalanobis = (hs - mean).pow(2) / (var + self.eps)

        if self.reduce == "mean":
            return mahalanobis.mean()
        return mahalanobis.sum()

