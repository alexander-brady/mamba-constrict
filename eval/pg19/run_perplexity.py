"""
PG19 Perplexity Evaluation using PyTorch

Calculates perplexity on PG19 test split using sliding window approach.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from finetune.data import collate_fn

# Load model configurations
model_map = json.loads(open("../config/model2path.json", encoding="utf-8").read())


def calculate_perplexity(model, dataloader, context_length, device, stride=512):
    """
    Calculate perplexity using sliding window approach.

    This implementation follows the standard methodology:
    - Uses overlapping windows with configurable stride
    - Only computes loss on new tokens in each window (via masking)
    - Accumulates weighted NLL by number of tokens
    """
    model.eval()
    nll_sum = 0.0
    n_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Context {context_length}"):
            input_ids = batch["input_ids"].to(device)

            # Process each sequence in batch
            for seq_idx in range(input_ids.size(0)):
                labels = input_ids[seq_idx : seq_idx + 1]
                seq_len = labels.size(1)

                prev_end_loc = 0
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + context_length, seq_len)
                    trg_len = end_loc - prev_end_loc

                    # Extract window
                    input_ids_window = labels[:, begin_loc:end_loc]
                    target_ids = input_ids_window.clone()

                    # Mask tokens from previous window
                    target_ids[:, :-trg_len] = -100

                    # Forward pass with labels
                    outputs = model(input_ids_window, labels=target_ids)

                    # outputs.loss is averaged over valid labels
                    # Model internally shifts labels left by 1
                    neg_log_likelihood = outputs.loss

                    # Count tokens that contributed to loss
                    num_valid_tokens = (target_ids != -100).sum().item()
                    batch_size = target_ids.size(0)
                    num_loss_tokens = (
                        num_valid_tokens - batch_size
                    )  # account for internal shift

                    # Accumulate weighted by number of tokens
                    nll_sum += neg_log_likelihood * num_loss_tokens
                    n_tokens += num_loss_tokens

                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break

    # Calculate perplexity
    avg_nll = nll_sum / n_tokens if n_tokens > 0 else float("inf")
    ppl = torch.exp(avg_nll).item()

    return {
        "perplexity": float(ppl),
        "total_tokens": int(n_tokens),
        "context_length": context_length,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity on PG19")
    parser.add_argument("--model", "-m", required=True, help="Model name from config")
    parser.add_argument(
        "--data_dir", "-d", required=True, help="Path to prepared test data"
    )
    parser.add_argument("--save_dir", "-s", default="results", help="Results directory")
    parser.add_argument(
        "--context_lengths",
        "-c",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    )
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_map.get(args.model, args.model)

    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Stride: {args.stride}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Load data
    test_data = load_from_disk(args.data_dir)
    test_data.set_format(type="torch", columns=["input_ids", "labels"])
    dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Prepare output
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, f"{args.model.replace('/', '_')}.jsonl")

    # Load existing results
    existing = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                existing[r["context_length"]] = r

    # Evaluate
    results = []
    with open(out_file, "w") as f:
        for ctx_len in args.context_lengths:
            # Skip if already done
            if ctx_len in existing:
                print(f"\nSkipping context length {ctx_len} (already evaluated)")
                results.append(existing[ctx_len])
                f.write(json.dumps(existing[ctx_len]) + "\n")
                f.flush()
                continue

            print(f"\n{'=' * 60}")
            print(f"Evaluating context length: {ctx_len}")
            print("=" * 60)

            result = calculate_perplexity(
                model, dataloader, ctx_len, device, stride=args.stride
            )
            result["model"] = args.model

            print(f"Perplexity: {result['perplexity']:.2f}")
            print(f"Tokens: {result['total_tokens']:,}")

            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Context':<12} {'Perplexity':<12} {'Tokens':<12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['context_length']:<12} {r['perplexity']:<12.2f} {r['total_tokens']:<12,}"
        )
    print("=" * 60)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
