import logging
import os

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

from .data import load_dataloader
from .lightning_module import FineTuner

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def finetune(cfg: DictConfig):
    # Get local rank for distributed training
    rank = int(os.environ["SLURM_LOCALID"])
    logger.info(f"Local rank: {rank}")

    # Log Hydra working directory
    hydra_wd = HydraConfig.get().runtime.output_dir
    if rank == 0:
        logger.info(f"Hydra working directory: {hydra_wd}")

    # Set seed for reproducibility
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    # Instantiate model and prepare for fine-tuning
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, low_cpu_mem_usage=True)
    model.train()
    model.gradient_checkpointing_enable()

    # Load dataset
    train_loader = load_dataloader(
        cfg.dataloader, split="train", save_dir=cfg.data["train"].save_dir
    )
    val_loader = load_dataloader(
        cfg.dataloader, split="validation", save_dir=cfg.data["validation"].save_dir
    )

    # Prepare loggers
    loggers = [
        # Log metrics to Weights & Biases
        WandbLogger(
            name=cfg.wandb.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            save_dir=hydra_wd,
            config=OmegaConf.to_container(cfg, resolve=True),
        ),
    ]

    # Instantiate fine-tuner and trainer
    fine_tuner = FineTuner(cfg, model)
    trainer = L.Trainer(
        default_root_dir=hydra_wd,
        logger=loggers,
        plugins=[SLURMEnvironment()],
        **cfg.trainer,
    )

    # Fine-tuning
    trainer.fit(fine_tuner, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Build save path: models/<task>/<run_id>
    task = "base" if cfg.data.name == "monology/pile-uncopyrighted" else cfg.data.name
    save_path = f"{cfg.model_dir}/{task}/{cfg.run_id}"

    # Save the fine-tuned model
    model.save_pretrained(
        save_path, 
        is_main_process=(rank == 0),
        # push_to_hub=cfg.push_to_hub,
        # repo_id=cfg.hub_repo_id
    )
    logger.info(f"Model saved to {save_path}")
