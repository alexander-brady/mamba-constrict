#!/usr/bin/env python3
"""
Smoke test script for vLLM with Mamba models.
This script performs basic inference to verify vLLM is working correctly.
"""

import argparse
import sys
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Test vLLM inference with Mamba model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Test prompt for inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Smoke Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    try:
        # Initialize the LLM
        print("\n[1/3] Loading model with vLLM...")
        llm = LLM(
            model=args.model,
            trust_remote_code=True,  # Required for Mamba models
            dtype="bfloat16",
        )
        print("✓ Model loaded successfully")

        # Set up sampling parameters
        print("\n[2/3] Setting up sampling parameters...")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        print("✓ Sampling parameters configured")

        # Run inference
        print(f"\n[3/3] Running inference...")
        print(f"Input: {args.prompt}")
        
        outputs = llm.generate([args.prompt], sampling_params)
        
        # Extract and print the result
        generated_text = outputs[0].outputs[0].text
        print(f"\n✓ Inference completed successfully!")
        print(f"\nGenerated output:")
        print("-" * 60)
        print(generated_text)
        print("-" * 60)
        
        print("\n" + "=" * 60)
        print("✓ vLLM smoke test PASSED")
        print("=" * 60)
        
        return 0

    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("✗ vLLM smoke test FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

