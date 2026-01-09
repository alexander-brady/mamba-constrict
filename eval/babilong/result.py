import argparse
import os
from collections import defaultdict

import pandas as pd


def analyze_model_results(model_dir):
    """Analyze BABILong results for a single model."""
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))

    if not os.path.isdir(model_dir):
        return None

    for filename in os.listdir(model_dir):
        if not filename.endswith(".csv"):
            continue

        # Parse filename: task_length_promptconfig.csv
        parts = filename.replace(".csv", "").split("_")
        if len(parts) < 2:
            continue

        task = parts[0]
        length = parts[1]

        filepath = os.path.join(model_dir, filename)
        try:
            df = pd.read_csv(filepath)
            if "target" not in df.columns or "output" not in df.columns:
                continue

            # Calculate accuracy (target word in output, case-insensitive)
            for _, row in df.iterrows():
                target = str(row["target"]).lower()
                output = str(row["output"]).lower()
                is_correct = target in output

                stats[task][length]["total"] += 1
                if is_correct:
                    stats[task][length]["correct"] += 1

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    return dict(stats) if stats else None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and display BABILong results"
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

    # Analyze results - look for babilong/ subdirectory in results_dir
    babilong_dir = os.path.join(args.results_dir, "babilong")
    model_name = args.model_name or os.path.basename(args.results_dir)

    model_data = analyze_model_results(babilong_dir)
    if not model_data:
        print(f"No results found in '{babilong_dir}'")
        return
    results_by_model = {model_name: model_data}

    # Collect all unique tasks and lengths
    all_tasks = set()
    all_lengths = set()
    for model_data in results_by_model.values():
        for task, lengths in model_data.items():
            all_tasks.add(task)
            all_lengths.update(lengths.keys())

    # Sort tasks and lengths
    sorted_tasks = sorted(all_tasks, key=lambda x: int(x.replace("qa", "")))
    length_order = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M"]
    sorted_lengths = [l for l in length_order if l in all_lengths]

    output_lines = []

    for model_name in sorted(results_by_model.keys()):
        model_data = results_by_model[model_name]

        output_lines.append(f"\n{'=' * 120}")
        output_lines.append(f"MODEL: {model_name}")
        output_lines.append("=" * 120)

        # Header row
        header = f"{'Task':<8}" + "".join(f"{length:>10}" for length in sorted_lengths) + f"{'Overall':>10}"
        output_lines.append(header)
        output_lines.append("-" * 120)

        # Calculate overall stats
        overall_correct = 0
        overall_total = 0

        # Task rows
        for task in sorted_tasks:
            if task not in model_data:
                continue

            row = f"{task:<8}"
            task_correct = 0
            task_total = 0

            for length in sorted_lengths:
                if length in model_data[task]:
                    stats = model_data[task][length]
                    acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    row += f"{acc:>9.1f}%"
                    task_correct += stats["correct"]
                    task_total += stats["total"]
                else:
                    row += f"{'-':>10}"

            # Task overall
            task_acc = (task_correct / task_total * 100) if task_total > 0 else 0
            row += f"{task_acc:>9.1f}%"
            output_lines.append(row)

            overall_correct += task_correct
            overall_total += task_total

        # Overall row
        output_lines.append("-" * 120)
        overall_row = f"{'OVERALL':<8}"

        for length in sorted_lengths:
            length_correct = 0
            length_total = 0
            for task in sorted_tasks:
                if task in model_data and length in model_data[task]:
                    stats = model_data[task][length]
                    length_correct += stats["correct"]
                    length_total += stats["total"]
            acc = (length_correct / length_total * 100) if length_total > 0 else 0
            overall_row += f"{acc:>9.1f}%"

        total_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
        overall_row += f"{total_acc:>9.1f}%"
        output_lines.append(overall_row)
        output_lines.append("=" * 120)

    # Print to console
    print("\n" + "=" * 120)
    print("BABILONG EVALUATION RESULTS")
    print("=" * 120)
    for line in output_lines:
        print(line)

    # Save to file
    result_file = os.path.join(args.results_dir, "babilong.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("BABILONG EVALUATION RESULTS\n")
        f.write("\n".join(output_lines))

    print(f"\nResults written to: {result_file}\n")


if __name__ == "__main__":
    main()
