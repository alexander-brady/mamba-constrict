#!/usr/bin/env python3
import argparse
from pathlib import Path
from transformers import AutoModel


def push_model(local_dir: Path, repo_id: str, private: bool):
    model = AutoModel.from_pretrained(local_dir)

    commit_msg = f"Add model from {local_dir.name}"
    print(f"Pushing {local_dir} → {repo_id}")

    model.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_msg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        required=True,
        help="List of HF repo ids (e.g. username/model-a username/model-b)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of local model directories (same order as --repos)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create repos as private (default: False)",
    )

    args = parser.parse_args()

    if len(args.repos) != len(args.models):
        raise ValueError("--repos and --models must have the same length")

    for model_dir, repo_id in zip(args.models, args.repos):
        push_model(Path(model_dir), repo_id, args.private)


if __name__ == "__main__":
    main()