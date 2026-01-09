import argparse
import json
import os
from collections import defaultdict


def analyze_model_results(filepath):
    """Analyze passkey retrieval results for a single model."""
    if not os.path.exists(filepath):
        return None

    results = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        return None

    # Aggregate by token length
    stats_by_length = defaultdict(lambda: {"total": 0, "correct": 0})

    for result in results:
        target_tokens = result["target_tokens"]
        stats_by_length[target_tokens]["total"] += 1
        if result["is_correct"]:
            stats_by_length[target_tokens]["correct"] += 1

    # Calculate accuracy for each length
    accuracies = {}
    for target_tokens in stats_by_length:
        stats = stats_by_length[target_tokens]
        accuracies[target_tokens] = (
            (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )

    return {"stats": dict(stats_by_length), "accuracies": accuracies}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and display passkey retrieval results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to analyze (if not provided, analyzes all models)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return

    # Analyze results - look for passkey.jsonl in results_dir
    filepath = os.path.join(args.results_dir, "passkey.jsonl")
    model_name = args.model_name or os.path.basename(args.results_dir)

    model_data = analyze_model_results(filepath)
    if not model_data:
        print(f"No results found in '{filepath}'")
        return
    results_by_model = {model_name: model_data}

    # Get all unique token lengths across all models and sort them
    all_lengths = set()
    for model_data in results_by_model.values():
        all_lengths.update(model_data["accuracies"].keys())
    sorted_lengths = sorted(all_lengths)

    # Build header
    header_parts = ["Model", "Overall"] + [
        f"{length:,} tokens" for length in sorted_lengths
    ]
    header = "\t".join(header_parts)

    # Build output lines
    output_lines = [header]

    for model_name in sorted(results_by_model.keys()):
        model_data = results_by_model[model_name]
        accuracies = model_data["accuracies"]
        stats = model_data["stats"]

        # Calculate overall accuracy
        total_correct = sum(s["correct"] for s in stats.values())
        total_tests = sum(s["total"] for s in stats.values())
        overall_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0

        # Build row
        row_parts = [model_name, f"{overall_acc:.1f}"]
        for length in sorted_lengths:
            if length in accuracies:
                row_parts.append(f"{accuracies[length]:.1f}")
            else:
                row_parts.append("-")

        output_lines.append("\t".join(row_parts))

    # Print to console
    print("\n" + "=" * 100)
    print("PASSKEY RETRIEVAL TEST RESULTS")
    print("=" * 100)
    for line in output_lines:
        print(line)
    print("=" * 100 + "\n")

    # Save to file
    result_file = os.path.join(args.results_dir, "passkey.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Results written to: {result_file}\n")


if __name__ == "__main__":
    main()
