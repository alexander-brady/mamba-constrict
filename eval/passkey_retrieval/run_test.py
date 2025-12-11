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
import time

import tiktoken
from numpy import random
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

# Load configuration files
model_map = json.loads(open("../config/model2path.json", encoding="utf-8").read())
maxlen_map = json.loads(open("../config/model2maxlen.json", encoding="utf-8").read())

URL = os.getenv("VLLM_URL")
API_KEY = os.getenv("VLLM_API_KEY")


def generate_prompt(n_garbage, seed=None):
    """Generates a text file and inserts an execute line at a random position."""
    if seed is not None:
        random.seed(seed)

    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 2000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = (
        f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    )
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    return "\n".join(lines), pass_key


def query_llm(
    prompt,
    model,
    tokenizer,
    client=None,
    temperature=0.0,
    max_new_tokens=10,
):
    """Query the LLM via vLLM API."""
    max_len = maxlen_map.get(model, 128000)
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
            prompt = tokenizer.decode(input_ids)

    tries = 0
    model_name = model_map[model] if model in model_map else model
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f'Error Occurs: "{str(e)}"        Retry ...')
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ""


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
    model = args.model
    n_values = args.n_values
    num_tests = args.num_tests

    # Setup tokenizer
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_map[model], trust_remote_code=True
        )

    # Setup OpenAI client for vLLM
    client = OpenAI(base_url=URL, api_key=API_KEY)

    # Prepare output directory and file
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, f"{model.split('/')[-1]}.jsonl")

    # Load existing results if any
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = f"{item['n_garbage']}_{item['test_id']}"
                has_data[key] = item

    results = []
    with open(out_file, "w", encoding="utf-8") as fout:
        for n in tqdm(n_values, desc="Context lengths"):
            for i in tqdm(range(num_tests), desc=f"Tests (n={n})", leave=False):
                key = f"{n}_{i}"

                # Skip if already processed
                if key in has_data:
                    result = has_data[key]
                    results.append(result)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    continue

                # Generate prompt and get expected passkey
                prompt_text, expected_passkey = generate_prompt(n, seed=i)

                # Get number of tokens
                num_tokens = len(tokenizer.encode(prompt_text))

                # Query the model
                response = query_llm(
                    prompt_text,
                    model,
                    tokenizer,
                    client,
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
                    "n_garbage": n,
                    "num_tokens": num_tokens,
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
    parser = argparse.ArgumentParser(description="Passkey Retrieval Test with vLLM")
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
        help="Model name (key from model2path.json)",
    )
    parser.add_argument(
        "--n_values",
        "-n",
        type=int,
        nargs="+",
        default=[
            0,
            100,
            500,
            1000,
            5000,
            8000,
            10000,
            12000,
            14000,
            18000,
            20000,
            25000,
            38000,
        ],
        help="List of garbage text lengths to test",
    )
    parser.add_argument(
        "--num_tests",
        "-t",
        type=int,
        default=50,
        help="Number of tests per garbage length",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Passkey Retrieval Test Configuration")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Garbage Lengths: {args.n_values}")
    print(f"Tests per Length: {args.num_tests}")
    print(f"vLLM URL: {URL}")
    print("=" * 80)

    # Run the test
    results, out_file = run_passkey_test(args)

    print("\n" + "=" * 80)
    print(f"Results saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
