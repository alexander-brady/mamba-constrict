"""
PG19 Perplexity Evaluation using PyTorch

Calculates perplexity on PG19 test split using sliding window approach.
Loads raw text files and processes each document individually.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity_on_document(
    model, tokenizer, text, context_length, device, stride=512
):
    """
    Calculate perplexity on a single document using sliding window.

    Args:
        model: The language model
        tokenizer: Tokenizer
        text: Raw text of the document
        context_length: Maximum context window size
        device: Device to run on
        stride: Stride for sliding window

    Returns:
        Tuple of (nll_sum, n_tokens) for this document
    """
    # Tokenize the entire document
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids.to(device)

    seq_len = input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0

    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + context_length, seq_len)
        trg_len = end_loc - prev_end_loc

        # Extract window
        input_ids_window = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_window.clone()

        # Mask tokens from previous window (only compute loss on new tokens)
        target_ids[:, :-trg_len] = -100

        # Forward pass with labels
        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)

        # outputs.loss is averaged over valid labels
        neg_log_likelihood = outputs.loss

        # Count tokens that contributed to loss
        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # account for internal shift

        # Accumulate weighted by number of tokens
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return nll_sum, n_tokens


def calculate_perplexity(
    model, tokenizer, documents, context_length, device, stride=512
):
    """
    Calculate perplexity across all documents using sliding window approach.

    This implementation follows the standard methodology:
    - Processes each document independently (no cross-document context)
    - Uses overlapping windows with configurable stride
    - Only computes loss on new tokens in each window (via masking)
    - Accumulates weighted NLL by number of tokens
    """
    model.eval()
    total_nll_sum = 0.0
    total_n_tokens = 0

    for text in tqdm(documents, desc=f"Context {context_length}"):
        nll_sum, n_tokens = calculate_perplexity_on_document(
            model, tokenizer, text, context_length, device, stride
        )
        total_nll_sum += nll_sum
        total_n_tokens += n_tokens

    # Calculate perplexity
    avg_nll = total_nll_sum / total_n_tokens if total_n_tokens > 0 else float("inf")
    ppl = torch.exp(avg_nll).item()

    return {
        "perplexity": float(ppl),
        "total_tokens": int(total_n_tokens),
        "context_length": context_length,
        "num_documents": len(documents),
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity on PG19")
    parser.add_argument(
        "--model", "-m", required=True, help="Model path (HuggingFace or local)"
    )
    parser.add_argument(
        "--data_dir",
        "-d",
        required=True,
        help="Path to raw PG19 test data (folder with .txt files)",
    )
    parser.add_argument("--save_dir", "-s", default="results", help="Results directory")
    parser.add_argument(
        "--context_lengths",
        "-c",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
    )
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Limit number of documents to process",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Stride: {args.stride}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Load raw text documents
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    text_files = sorted(list(data_path.glob("*.txt")))
    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")

    print(f"\nFound {len(text_files)} text files in {data_path}")

    # Load documents
    documents = []
    for txt_file in tqdm(text_files[: args.max_documents], desc="Loading documents"):
        with open(txt_file, encoding="utf-8") as f:
            text = f.read()
            if text.strip():  # Skip empty files
                documents.append(text)

    print(f"Loaded {len(documents)} non-empty documents")

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
                model, tokenizer, documents, ctx_len, device, stride=args.stride
            )
            result["model"] = args.model

            print(f"Perplexity: {result['perplexity']:.2f}")
            print(f"Tokens: {result['total_tokens']:,}")
            print(f"Documents: {result['num_documents']}")

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
