import logging

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

    # Build save path: models/model_name_criterion_dataset[_task]
    model_name = cfg.model.name.split("/")[-1]
    dataset_name = cfg.data.name.split("/")[-1]    
    criterion_name = cfg.loss._target_.split(".")[-1].lower()
    if cfg.loss.get("weight", None) is not None:
        criterion_name += f"_w{cfg.loss.weight}"    
    save_path = f"{cfg.model_dir}/{model_name}_{criterion_name}_{dataset_name}"
    if cfg.data.get("use_babilong", False) and cfg.data.get("task"):
        save_path += f"_{cfg.data.task}"

    # Save the fine-tuned model
    model.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")