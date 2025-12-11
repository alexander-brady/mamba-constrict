import itertools
import logging
from pathlib import Path
from typing import Literal

import datasets
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def prepare_data(
    data_cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    split: Literal["train", "validation", "test"],
):
    """
    Load, process, and save dataset split for fine-tuning.

    Args:
        data_cfg (DictConfig): Data configuration.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        split (Literal["train", "validation"]): Dataset split to process.
    """
    # Load dataset
    if data_cfg.get("use_local", False):
        # Load from local directory (e.g., downloaded Gutenberg/PG-19 data)
        # Each split folder contains multiple text files
        local_dir = Path(data_cfg[split].save_dir)
        logger.info(f"Loading {split} split from local directory: {local_dir}")

        if not local_dir.exists():
            raise FileNotFoundError(
                f"Local data directory not found: {local_dir}\n"
                f"Please run the download script first: ./scripts/download_gutenberg.sh"
            )

        # Get all text files in the directory
        text_files = list(local_dir.glob("*.txt"))
        if not text_files:
            raise FileNotFoundError(
                f"No text files found in {local_dir}. "
                f"Please ensure the data has been downloaded correctly."
            )

        logger.info(f"Found {len(text_files)} text files in {local_dir}")

        # Load all text files as a dataset
        # "text" = use the text file loader
        # data_files key can be anything, we use the actual split name for clarity
        # split parameter must match the data_files key
        ds = datasets.load_dataset(
            "text",
            data_files={split: [str(f) for f in text_files]},
            split=split,
        )
        assert isinstance(ds, datasets.Dataset)  # for type checker

        logger.info(f"Loaded {split} split with {len(ds)} samples from local files.")
    else:
        # Load from HuggingFace Hub
        ds = datasets.load_dataset(
            data_cfg.name, split=data_cfg[split].split, trust_remote_code=True
        )
        assert isinstance(ds, datasets.Dataset)  # for type checker
        logger.info(f"Loaded {split} split with {len(ds)} samples from HuggingFace.")

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=False,
            return_attention_mask=False,
        )

        # append eos for each sequence
        input_ids = [ids + [tokenizer.eos_token_id] for ids in enc["input_ids"]]

        return {"input_ids": input_ids}

    # Apply tokenization
    ds = ds.map(
        tokenize,
        remove_columns=ds.column_names,
        batched=True,
        batch_size=10000,
        num_proc=8,
    )

    block_size = data_cfg.block_size

    def pack_samples(samples):
        # concatenate then chunk
        ids = list(itertools.chain.from_iterable(samples["input_ids"]))

        # Trim to a multiple of block_size
        total_len = (len(ids) // block_size) * block_size
        ids = ids[:total_len]

        # Create chunks
        flat = ids[:total_len]
        chunks = [flat[i : i + block_size] for i in range(0, total_len, block_size)]

        return {"input_ids": chunks, "labels": chunks}

    # Pack samples into fixed-size blocks
    ds = ds.map(pack_samples, batched=True, batch_size=10000, num_proc=8)

    # Save processed dataset
    output_dir = Path(data_cfg[split].save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds.save_to_disk(output_dir)
    logger.info(f"Saved dataset for {split} to {output_dir}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        logger.warning("Added missing EOS token to tokenizer.")

    # Save processed data for train, validation, and test splits
    prepare_data(cfg.data, tokenizer, split="train")
    prepare_data(cfg.data, tokenizer, split="validation")

    # Prepare test split if it exists in config
    if "test" in cfg.data:
        prepare_data(cfg.data, tokenizer, split="test")


if __name__ == "__main__":
    main()  # type: ignore
