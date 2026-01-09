"""
Plot PG19 Perplexity Results

Creates visualizations of perplexity vs context length, similar to the GovReport benchmark.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_model_results(filepath):
    """Load results for a single model."""
    if not os.path.exists(filepath):
        return None

    results = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        return None

    # Sort by context length
    results.sort(key=lambda x: x["context_length"])
    return results


def load_results(results_dir, model_name=None):
    """Load result files from the results directory."""
    results_by_model = {}

    # Look for perplexity.jsonl in results_dir
    filepath = os.path.join(results_dir, "perplexity.jsonl")
    if not model_name:
        model_name = os.path.basename(results_dir)

    results = load_model_results(filepath)
    if results:
        results_by_model[model_name] = results

    return results_by_model


def plot_single_model(model_name, results, output_path, title_prefix="PG19"):
    """
    Plot perplexity vs context length for a single model.
    Similar to the GovReport visualization style.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    context_lengths = [r["context_length"] for r in results]
    perplexities = [r["perplexity"] for r in results]

    # Plot with markers
    ax.plot(
        context_lengths,
        perplexities,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#d62728",  # Red color similar to the image
    )

    # Set log scale for y-axis
    ax.set_yscale("log")

    # Set x-axis to powers of 2
    # Find appropriate power of 2 ticks
    min_ctx = min(context_lengths)
    max_ctx = max(context_lengths)
    min_pow = int(np.floor(np.log2(min_ctx)))
    max_pow = int(np.ceil(np.log2(max_ctx)))

    x_ticks = [2**i for i in range(min_pow, max_pow + 1)]
    x_tick_labels = [f"$2^{{{i}}}$" for i in range(min_pow, max_pow + 1)]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xscale("log", base=2)

    # Labels and title
    ax.set_xlabel("Context Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("PPL", fontsize=12, fontweight="bold")

    # Extract model name for title (e.g., "Mamba-1.4B" from path)
    model_display_name = model_name.split("/")[-1] if "/" in model_name else model_name
    ax.set_title(
        f"{title_prefix}\n{model_display_name}", fontsize=14, fontweight="bold"
    )

    # Grid
    ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def plot_multiple_models(
    results_by_model, output_path, title="PG19 Perplexity Comparison"
):
    """
    Plot perplexity vs context length for multiple models on the same plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_model)))

    for idx, (model_name, results) in enumerate(sorted(results_by_model.items())):
        # Extract data
        context_lengths = [r["context_length"] for r in results]
        perplexities = [r["perplexity"] for r in results]

        # Get display name
        model_display_name = (
            model_name.split("/")[-1] if "/" in model_name else model_name
        )

        # Plot
        ax.plot(
            context_lengths,
            perplexities,
            marker="o",
            linewidth=2,
            markersize=8,
            color=colors[idx],
            label=model_display_name,
        )

    # Set log scale for y-axis
    ax.set_yscale("log")

    # Set x-axis to powers of 2
    all_ctx = []
    for results in results_by_model.values():
        all_ctx.extend([r["context_length"] for r in results])

    min_ctx = min(all_ctx)
    max_ctx = max(all_ctx)
    min_pow = int(np.floor(np.log2(min_ctx)))
    max_pow = int(np.ceil(np.log2(max_ctx)))

    x_ticks = [2**i for i in range(min_pow, max_pow + 1)]
    x_tick_labels = [f"$2^{{{i}}}$" for i in range(min_pow, max_pow + 1)]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xscale("log", base=2)

    # Labels and title
    ax.set_xlabel("Context Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("PPL", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    ax.legend(loc="best", fontsize=10)

    # Grid
    ax.grid(True, which="both", alpha=0.3, linestyle="-", linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def create_summary_table(results_by_model, output_path):
    """Create a summary table of results."""
    with open(output_path, "w") as f:
        # Get all unique context lengths
        all_ctx_lengths = set()
        for results in results_by_model.values():
            all_ctx_lengths.update([r["context_length"] for r in results])
        ctx_lengths = sorted(all_ctx_lengths)

        # Write header
        f.write("Model\t" + "\t".join([str(ctx) for ctx in ctx_lengths]) + "\n")

        # Write data for each model
        for model_name, results in sorted(results_by_model.items()):
            # Create dict for easy lookup
            ppl_by_ctx = {r["context_length"]: r["perplexity"] for r in results}

            # Get display name
            model_display_name = (
                model_name.split("/")[-1] if "/" in model_name else model_name
            )

            # Write row
            row = [model_display_name]
            for ctx in ctx_lengths:
                if ctx in ppl_by_ctx:
                    row.append(f"{ppl_by_ctx[ctx]:.2f}")
                else:
                    row.append("-")
            f.write("\t".join(row) + "\n")

    print(f"Saved summary table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PG19 perplexity results")
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save plots (defaults to results_dir)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="PG19",
        help="Title prefix for plots",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default=None,
        help="Model name to analyze (if not provided, analyzes all models)",
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print("Loading results...")
    results_by_model = load_results(args.results_dir, args.model_name)

    if not results_by_model:
        if args.model_name:
            print(f"No results found for model '{args.model_name}'")
        else:
            print(f"No result files found in '{args.results_dir}'")
        return

    print(f"Found results for {len(results_by_model)} model(s)")

    # Create plots for each model
    for model_name, results in results_by_model.items():
        output_path = os.path.join(output_dir, "perplexity.png")
        plot_single_model(model_name, results, output_path, title_prefix=args.title)

    # Create summary table
    summary_path = os.path.join(output_dir, "perplexity.txt")
    create_summary_table(results_by_model, summary_path)

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
