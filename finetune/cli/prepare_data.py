import itertools
import logging
import random
import sys
from pathlib import Path
from typing import Literal

import datasets
import hydra
import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

# Add eval directory to path to import passkey_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval" / "passkey_retrieval"))
from passkey_utils import TASK_DESCRIPTION, GARBAGE_SENTENCE, FINAL_QUESTION, generate_passkey_info

logger = logging.getLogger(__name__)


def generate_passkey_sample(tokenizer: PreTrainedTokenizer, target_length: int, depth: float, seed: int):
    """
    Generate a single passkey retrieval sample.

    Args:
        tokenizer: Tokenizer to use for encoding
        target_length: Target sequence length in tokens
        depth: Position of passkey (0.0 = start, 1.0 = end)
        seed: Random seed for reproducibility

    Returns:
        dict with input_ids and labels (labels mask everything except passkey answer)
    """
    rng = random.Random(seed)

    # Generate random passkey
    passkey = rng.randint(1, 50000)

    # Create the task components using shared utilities
    information_line = generate_passkey_info(passkey)

    # Tokenize the key components to know their lengths
    task_tokens = tokenizer.encode(TASK_DESCRIPTION, add_special_tokens=False)
    info_tokens = tokenizer.encode(information_line, add_special_tokens=False)
    question_tokens = tokenizer.encode(FINAL_QUESTION, add_special_tokens=False)
    answer_tokens = tokenizer.encode(f" {passkey}", add_special_tokens=False)

    # Calculate how many garbage tokens we need
    fixed_tokens = len(task_tokens) + len(info_tokens) + len(question_tokens) + len(answer_tokens)
    garbage_tokens_needed = target_length - fixed_tokens - 1  # -1 for EOS

    if garbage_tokens_needed < 0:
        raise ValueError(f"Target length {target_length} is too short for passkey task")

    # Calculate position of passkey based on depth
    # depth=0.0 means passkey right after task description
    # depth=1.0 means passkey right before question
    passkey_position = int(depth * garbage_tokens_needed)

    # Split garbage into prefix and suffix
    garbage_prefix_tokens_needed = passkey_position
    garbage_suffix_tokens_needed = garbage_tokens_needed - passkey_position

    # Generate garbage tokens by repeating the garbage sentence
    garbage_sentence_tokens = tokenizer.encode(GARBAGE_SENTENCE, add_special_tokens=False)

    def generate_garbage_tokens(n_tokens):
        if n_tokens == 0:
            return []
        tokens = []
        while len(tokens) < n_tokens:
            tokens.extend(garbage_sentence_tokens)
        return tokens[:n_tokens]

    garbage_prefix = generate_garbage_tokens(garbage_prefix_tokens_needed)
    garbage_suffix = generate_garbage_tokens(garbage_suffix_tokens_needed)

    # Assemble the full sequence
    input_ids = (
        task_tokens +
        garbage_prefix +
        info_tokens +
        garbage_suffix +
        question_tokens +
        answer_tokens +
        [tokenizer.eos_token_id]
    )

    # Create labels: mask everything except the answer tokens
    # Use -100 for masked tokens (ignored by loss function)
    labels = [-100] * len(input_ids)

    # Only keep the answer tokens (the passkey number)
    answer_start = len(task_tokens) + len(garbage_prefix) + len(info_tokens) + len(garbage_suffix) + len(question_tokens)
    for i in range(len(answer_tokens)):
        labels[answer_start + i] = input_ids[answer_start + i]

    # Also keep EOS token
    labels[-1] = tokenizer.eos_token_id

    return {
        "input_ids": input_ids,
        "labels": labels,
        "passkey": passkey,
        "depth": depth,
    }


def prepare_passkey_data(
    data_cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    split: Literal["train", "validation", "test"],
):
    """Prepare passkey retrieval dataset."""
    num_samples = data_cfg[split].num_samples
    block_size = data_cfg.block_size
    min_depth = data_cfg.get("min_depth", 0.0)
    max_depth = data_cfg.get("max_depth", 1.0)

    logger.info(f"Generating {num_samples} passkey samples for {split} split")
    logger.info(f"Target length: {block_size} tokens")
    logger.info(f"Depth range: [{min_depth}, {max_depth}]")

    samples = []
    for i in range(num_samples):
        # Sample a random depth for each sample
        depth = random.uniform(min_depth, max_depth)

        # Generate sample with unique seed
        sample = generate_passkey_sample(
            tokenizer=tokenizer,
            target_length=block_size,
            depth=depth,
            seed=i + hash(split),  # Different seeds for different splits
        )
        samples.append(sample)

    # Create dataset from samples
    ds = datasets.Dataset.from_dict({
        "input_ids": [s["input_ids"] for s in samples],
        "labels": [s["labels"] for s in samples],
        "passkey": [s["passkey"] for s in samples],
        "depth": [s["depth"] for s in samples],
    })

    # Save processed dataset
    output_dir = Path(data_cfg[split].save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds.save_to_disk(output_dir)
    logger.info(f"Saved {len(ds)} passkey samples to {output_dir}")


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
    # Check if this is a passkey dataset
    if data_cfg.get("use_passkey", False):
        prepare_passkey_data(data_cfg, tokenizer, split)
        return

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

    block_size = data_cfg.block_size
    use_truncation = data_cfg[split].get("use_truncation", False)

    if use_truncation:
        # Truncate/pad each sample to fixed block_size
        logger.info(f"Using truncation mode: each sample will be exactly {block_size} tokens")

        def tokenize(batch):
            # Tokenize with truncation and padding to fixed length
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_attention_mask=False,
            )

            # Labels are the same as input_ids for causal language modeling
            return {"input_ids": enc["input_ids"], "labels": enc["input_ids"]}

        # Apply tokenization
        ds = ds.map(
            tokenize,
            remove_columns=ds.column_names,
            batched=True,
            batch_size=10000,
            num_proc=8,
        )
    else:
        # Pack multiple samples into fixed-size blocks (original behavior)
        logger.info(f"Using packing mode: concatenating and chunking into {block_size} token blocks")

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



if __name__ == "__main__":
    main()  # type: ignore
