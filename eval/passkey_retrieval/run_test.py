# This is a modification of the original code
# https://github.com/epfml/landmark-attention/blob/111ee30e693ccc23a12b57c1d41f8ae2cc5b4867/llama/run_test.py#L96
# The original code license:
# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from passkey_utils import generate_prompt_random_depth


def calc_str_length(token_length, letters_per_token=3.65):
    """Calculate approximate character length from token length."""
    return int(token_length * letters_per_token)


def generate_prompt(n_garbage, seed=None):
    """Generates a text file and inserts an execute line at a random position."""
    return generate_prompt_random_depth(n_garbage, seed)


def query_llm(
    prompt,
    model,
    tokenizer,
    device,
    temperature=0.0,
    max_new_tokens=10,
):
    """Query the LLM using local model."""
    # Tokenize prompt (no truncation for SSMs)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate response
    # use_cache=False to avoid memory issues with long contexts
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=False,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
    return response


def extract_passkey(response):
    """Extract the passkey from model response."""
    try:
        # Try to find the first number in the response
        match = re.search(r"\d+", response)
        if match:
            return int(match.group())
        else:
            return None
    except Exception:
        return None


def run_passkey_test(args):
    """Run passkey retrieval test."""
    model_name = args.model
    token_lengths = args.token_lengths
    num_tests = args.num_tests

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading model from: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # Prepare output directory and file
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, f"{model_name.replace('/', '_')}.jsonl")

    # Load existing results if any
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = f"{item['target_tokens']}_{item['test_id']}"
                has_data[key] = item

    results = []
    with open(out_file, "w", encoding="utf-8") as fout:
        for target_tokens in tqdm(token_lengths, desc="Token lengths"):
            # Convert token length to character length
            str_length = calc_str_length(target_tokens)

            for i in tqdm(range(num_tests), desc=f"Tests (tokens={target_tokens})", leave=False):
                key = f"{target_tokens}_{i}"

                # Skip if already processed
                if key in has_data:
                    result = has_data[key]
                    results.append(result)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    continue

                # Generate prompt and get expected passkey
                prompt_text, expected_passkey = generate_prompt(str_length, seed=i)

                # Get actual number of tokens
                actual_tokens = len(tokenizer.encode(prompt_text))

                # Query the model
                response = query_llm(
                    prompt_text,
                    model,
                    tokenizer,
                    device,
                    temperature=0.0,
                    max_new_tokens=10,
                )

                # Extract passkey from response
                predicted_passkey = extract_passkey(response)

                # Check if correct
                is_correct = predicted_passkey == expected_passkey

                # Store result
                result = {
                    "test_id": i,
                    "target_tokens": target_tokens,
                    "actual_tokens": actual_tokens,
                    "str_length": str_length,
                    "expected_passkey": expected_passkey,
                    "predicted_passkey": predicted_passkey,
                    "response": response,
                    "is_correct": is_correct,
                }
                results.append(result)

                # Write to file
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    return results, out_file


def main():
    parser = argparse.ArgumentParser(
        description="Passkey Retrieval Test with Local Models"
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model path (HuggingFace or local)",
    )
    parser.add_argument(
        "--token_lengths",
        "-t",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        help="List of target token lengths to test",
    )
    parser.add_argument(
        "--num_tests",
        "-n",
        type=int,
        default=50,
        help="Number of tests per token length",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Passkey Retrieval Test Configuration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Token Lengths: {args.token_lengths}")
    print(f"Tests per Length: {args.num_tests}")
    print("=" * 80)

    # Run the test
    results, out_file = run_passkey_test(args)

    print("\n" + "=" * 80)
    print(f"Results saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
