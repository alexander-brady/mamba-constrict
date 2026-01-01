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

    def forward(self, **batch):
        backbone_out = self.model.backbone(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=False,
            return_dict=True,
        )

        hidden_states = backbone_out[0]  # (B, S, D)
        logits = self.model.lm_head(
            hidden_states.to(self.model.lm_head.weight.dtype)
        ).float()

        loss = None
        if "labels" in batch:
            loss = self.model.loss_function(
                logits=logits,
                labels=batch["labels"],
                vocab_size=self.model.config.vocab_size,
            )

        return hidden_states, logits, loss

    def training_step(self, batch: dict, batch_idx: int):
        # Forward pass with hidden states output
        hidden_states, logits, loss = self(**batch)

        # Compute auxiliary loss using hidden states
        aux_loss = self.auxiliary_loss(hidden_states)

        # Log losses
        log_dict = {
            "train/ce_loss": loss,
            "train/auxiliary_loss": aux_loss,
            "train/loss": loss + aux_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, sync_dist=True, batch_size=len(batch["input_ids"])
        )
        return loss + aux_loss

    def validation_step(self, batch: dict, batch_idx: int):
        with torch.inference_mode():
            hidden_states, logits, loss = self(**batch)

            accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
            aux_loss = self.auxiliary_loss(hidden_states)

        log_dict = {
            "val/ce_loss": loss,
            "val/auxiliary_loss": aux_loss,
            "val/loss": loss + aux_loss,
            "val/accuracy": accuracy,
        }
        self.log_dict(
            log_dict, prog_bar=True, sync_dist=True, batch_size=len(batch["input_ids"])
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, params=self.model.parameters())

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
