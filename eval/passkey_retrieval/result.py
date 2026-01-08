import argparse
import json
import os
from collections import defaultdict


def analyze_results(results_dir):
    """Analyze passkey retrieval results and aggregate by model."""
    results_by_model = {}

    # Process all result files
    for filename in os.listdir(results_dir):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(results_dir, filename)
        model_name = filename.replace(".jsonl", "")

        # Read results
        results = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))

        # Aggregate by token length
        stats_by_length = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )

        for result in results:
            target_tokens = result["target_tokens"]
            stats_by_length[target_tokens]["total"] += 1
            if result["is_correct"]:
                stats_by_length[target_tokens]["correct"] += 1

        # Calculate accuracy and average tokens for each length
        accuracies = {}
        for target_tokens in stats_by_length:
            stats = stats_by_length[target_tokens]
            accuracies[target_tokens] = (
                (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )

        results_by_model[model_name] = {
            "stats": stats_by_length,
            "accuracies": accuracies,
        }

    return results_by_model


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
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return

    # Analyze results
    results_by_model = analyze_results(args.results_dir)

    if not results_by_model:
        print(f"No result files found in '{args.results_dir}'")
        return

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
    result_file = os.path.join(args.results_dir, "result.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Results written to: {result_file}\n")


if __name__ == "__main__":
    main()
