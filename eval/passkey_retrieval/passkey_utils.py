"""Shared utilities for passkey retrieval tasks."""

import random

# Constants used in passkey tasks
TASK_DESCRIPTION = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
GARBAGE_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
FINAL_QUESTION = "What is the pass key? The pass key is"


def generate_passkey_info(passkey: int) -> str:
    """Generate the passkey information line."""
    return f"The pass key is {passkey}. Remember it. {passkey} is the pass key."


def generate_garbage_text(n_chars: int) -> str:
    """
    Generate garbage text of approximately n_chars length.

    Args:
        n_chars: Desired character length

    Returns:
        Garbage text string
    """
    if n_chars == 0:
        return ""

    garbage_text = GARBAGE_SENTENCE
    while len(garbage_text) < n_chars:
        garbage_text = " ".join([garbage_text] * 2)

    return garbage_text[:n_chars]


def generate_prompt_with_depth(
    n_garbage: int, depth: float, seed: int | None = None
) -> tuple[str, int]:
    """
    Generate a passkey prompt with specified depth.

    Args:
        n_garbage: Total character length of garbage text
        depth: Position of passkey (0.0 = start, 1.0 = end)
        seed: Random seed for passkey generation

    Returns:
        Tuple of (prompt_text, passkey)
    """
    if seed is not None:
        random.seed(seed)

    # Generate random passkey
    passkey = random.randint(1, 50000)

    # Calculate garbage split based on depth
    n_garbage_prefix = int(depth * n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    # Generate components
    garbage_prefix = generate_garbage_text(n_garbage_prefix)
    garbage_suffix = generate_garbage_text(n_garbage_suffix)
    information_line = generate_passkey_info(passkey)

    # Assemble prompt
    lines = [
        TASK_DESCRIPTION,
        garbage_prefix,
        information_line,
        garbage_suffix,
        FINAL_QUESTION,
    ]

    return "\n".join(lines), passkey


def generate_prompt_random_depth(
    n_garbage: int, seed: int | None = None
) -> tuple[str, int]:
    """
    Generate a passkey prompt with random depth.

    Args:
        n_garbage: Total character length of garbage text
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prompt_text, passkey)
    """
    if seed is not None:
        random.seed(seed)

    # Random depth
    depth = random.random()

    return generate_prompt_with_depth(n_garbage, depth, seed=None)
