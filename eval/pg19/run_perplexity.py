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
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity_on_document(
    model, tokenizer, text, context_length, device, stride=512,
    n_windows=10, last_k=100,
):
    """
    Calculate perplexity on a single document using DeciMamba-style evaluation.

    Args:
        model: The language model
        tokenizer: Tokenizer
        text: Raw text of the document
        context_length: Maximum context window size
        device: Device to run on
        stride: Stride for sliding window (unused, kept for compatibility)
        n_windows: Number of windows to evaluate (DeciMamba: 10)
        last_k: Number of last labels to score per window (DeciMamba: 100)

    Returns:
        Tuple of (nll_sum, n_tokens) for this document
    """
    # Tokenize the entire document
    # - Avoid add_special_tokens=True because different tokenizers add BOS/EOS differently,
    #   which changes PPL and hurts reproducibility.
    # - We manually add BOS token if available.
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([bos, input_ids], dim=1)

    input_ids = input_ids.to(device)

    seq_len = input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0

    # DeciMamba window selection: n_windows windows with maximal constant stride
    if seq_len <= context_length:
        begins = [0]  # only one window fits
    else:
        max_start = seq_len - context_length
        if n_windows <= 1:
            begins = [max_start]
        else:
            stride_dm = max(1, max_start // (n_windows - 1))
            begins = [i * stride_dm for i in range(n_windows)]
            # safety: ensure windows fit
            begins = [b for b in begins if b <= max_start]
            if not begins:
                begins = [0]

    for begin_loc in begins:
        end_loc = min(begin_loc + context_length, seq_len)

        # Extract window
        input_ids_window = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_window.clone()

        # DeciMamba: score only the last `last_k` labels
        keep_from = max(target_ids.size(1) - last_k, 0)
        target_ids[:, :keep_from] = -100

        # Forward pass with labels
        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)

        # outputs.loss is averaged over valid labels
        neg_log_likelihood = outputs.loss

        # HF causal LM loss internally shifts labels left by 1, i.e. it computes loss on labels[:, 1:].
        # So the correct number of loss terms is the count of unmasked labels in target_ids[:, 1:].
        num_loss_tokens = (target_ids[:, 1:] != -100).sum().item()

        # Accumulate weighted by number of tokens
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

    return nll_sum, n_tokens


def calculate_perplexity(
    model, tokenizer, documents, context_length, device, stride=512
):
    """
    Calculate perplexity across a list of documents using DeciMamba-style evaluation.

    DeciMamba protocol (per document):
      - Evaluate a fixed number of windows (default 10) of length `context_length`
        using a maximal constant stride so the windows span the document.
      - Compute loss only on the last `last_k` labels in each window (default 100),
        which approximates model performance near positions
        [context_length - last_k, context_length].

    Aggregation:
      - The model returns an average loss over the unmasked labels (after the internal
        causal shift). We convert this to total NLL by multiplying by the number of
        evaluated labels, sum across all windows and documents, then divide by the
        total number of evaluated labels.
      - Perplexity is exp(average_nll).

    Notes:
      - Each document is evaluated independently (no cross-document context).
      - The `stride` argument is kept for API compatibility but is not used in this
        DeciMamba-style evaluator.
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
        "--tokenizer", type=str, default=None, help="Tokenizer path (defaults to model path if not specified)"
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
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_dir", type=str, default=None, help="Weights & Biases output directory"
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_display_name = os.path.basename(args.model)

    # Initialize wandb
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or model_display_name,
            dir=args.wandb_dir,
            config={
                "model": args.model,
                "context_lengths": args.context_lengths,
                "stride": args.stride,
                "max_documents": args.max_documents,
            },
            job_type="perplexity",
        )

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Stride: {args.stride}")

    # Load model
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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
    out_file = os.path.join(args.save_dir, "perplexity.jsonl")

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

            # Log to wandb
            if use_wandb:
                wandb.log({
                    f"perplexity/{ctx_len}/ppl": result["perplexity"],
                    f"perplexity/{ctx_len}/tokens": result["total_tokens"],
                })

            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    # Finish wandb
    if use_wandb:
        wandb.finish()

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