"""
Utility for discovering and managing models for evaluation.

This module provides functions to:
1. Discover local finetuned models in the models/ directory
2. Load HuggingFace model paths from models/huggingface_models.txt
3. Generate model lists for evaluation scripts
"""

import os
from pathlib import Path
from typing import Dict, List


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # This file is in eval/, repo root is parent
    return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_repo_root() / "models"


def get_huggingface_models() -> List[str]:
    """
    Load HuggingFace model paths from models/huggingface_models.txt.

    Returns:
        List of HuggingFace model paths (e.g., ["state-spaces/mamba-130m-hf"])
    """
    hf_file = get_models_dir() / "huggingface_models.txt"

    if not hf_file.exists():
        return []

    models = []
    with open(hf_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                models.append(line)

    return models


def get_local_models() -> List[Path]:
    """
    Discover local finetuned models in the models/ directory.

    A directory is considered a model if it contains either:
    - config.json (HuggingFace model format)
    - pytorch_model.bin or model.safetensors

    Returns:
        List of paths to local model directories
    """
    models_dir = get_models_dir()

    if not models_dir.exists():
        return []

    local_models = []

    # Iterate over subdirectories in models/
    for item in models_dir.iterdir():
        if not item.is_dir():
            continue

        # Skip hidden directories
        if item.name.startswith('.'):
            continue

        # Check if it looks like a model directory
        has_config = (item / "config.json").exists()
        has_weights = (
            (item / "pytorch_model.bin").exists() or
            (item / "model.safetensors").exists() or
            any(item.glob("*.safetensors"))
        )

        if has_config or has_weights:
            local_models.append(item)

    return sorted(local_models)


def get_all_models() -> Dict[str, str]:
    """
    Get all models (both local and HuggingFace).

    Returns:
        Dictionary mapping model names to paths:
        - For HuggingFace models: {"mamba-130m-hf": "state-spaces/mamba-130m-hf"}
        - For local models: {"my_finetuned_model": "/path/to/models/my_finetuned_model"}
    """
    models = {}

    # Add HuggingFace models
    for hf_path in get_huggingface_models():
        # Use the model name (last part) as key
        model_name = hf_path.split('/')[-1]
        models[model_name] = hf_path

    # Add local models
    for local_path in get_local_models():
        # Use directory name as model name
        model_name = local_path.name
        models[model_name] = str(local_path.absolute())

    return models


def list_models() -> None:
    """Print all available models."""
    models = get_all_models()

    if not models:
        print("No models found.")
        print(f"Add HuggingFace models to: {get_models_dir() / 'huggingface_models.txt'}")
        print(f"Place local models in: {get_models_dir()}/")
        return

    print(f"Found {len(models)} model(s):")
    print()

    hf_models = get_huggingface_models()
    local_models = get_local_models()

    if hf_models:
        print("HuggingFace Models:")
        for hf_path in hf_models:
            model_name = hf_path.split('/')[-1]
            print(f"  {model_name:30} -> {hf_path}")
        print()

    if local_models:
        print("Local Models:")
        for local_path in local_models:
            print(f"  {local_path.name:30} -> {local_path}")
        print()


if __name__ == "__main__":
    list_models()
