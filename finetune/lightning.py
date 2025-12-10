import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import PreTrainedModel


class FineTuner(L.LightningModule):
    """Fine-tuning module for a pre-trained model."""

    def __init__(self, cfg: DictConfig, model: PreTrainedModel):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.auxiliary_loss = instantiate(self.cfg.loss)
        self.save_hyperparameters(cfg)
        
        # Hook to capture last hidden states
        self._last_hidden = None
        self.model.backbone.register_forward_hook(
            lambda module, inp, out: setattr(self, '_last_hidden', out.last_hidden_state)
        )

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int):
        # Forward pass with hidden states output
        outputs = self.model(**batch, return_dict=True)
        loss = outputs.loss

        # Compute auxiliary loss using hidden states
        auxiliary_loss = self.auxiliary_loss(
            batch["labels"], outputs.logits, self._last_hidden
        )

        # Log losses
        log_dict = {
            "train/ce_loss": loss,
            "train/auxiliary_loss": auxiliary_loss,
            "train/loss": loss + auxiliary_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, sync_dist=True, batch_size=len(batch["input_ids"])
        )
        return loss + auxiliary_loss

    def validation_step(self, batch: dict, batch_idx: int) -> dict[str, float]:
        with torch.inference_mode():
            outputs = self.model(**batch, return_dict=True)
            loss = outputs.loss
            auxiliary_loss = self.auxiliary_loss(
                batch["labels"], outputs.logits, self._last_hidden
            )
            accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()

        log_dict = {
            "val/ce_loss": loss,
            "val/auxiliary_loss": auxiliary_loss,
            "val/loss": loss + auxiliary_loss,
            "val/accuracy": accuracy,
        }
        self.log_dict(
            log_dict, prog_bar=True, sync_dist=True, batch_size=len(batch["input_ids"])
        )
        return log_dict

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, params=self.model.parameters())

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
