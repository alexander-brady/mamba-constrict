import argparse
import json
import os
import sys
from pathlib import Path

import datasets
import pandas as pd
import wandb
from prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def query_llm(prompt, llm, temperature=0.0, max_tokens=20):
    """Query the model with vLLM."""
    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    return response


def main(
    results_folder: str,
    model_name: str,
    model_path: str,
    tasks: list[str],
    split_names: list[str],
    dataset_name: str,
    use_instruction: bool,
    use_examples: bool,
    use_post_prompt: bool,
    tokenizer_path: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_dir: str | None = None,
) -> None:
    """
    Main function to get model predictions on babilong and save them using vLLM.

    Args:
        results_folder (str): Folder to store results.
        model_name (str): Name of the model (used to save model results)
        model_path (str): path to local model weights or HuggingFace model ID
        tasks (List[str]): List of tasks to evaluate.
        split_names (List[str]): List of lengths to evaluate.
        dataset_name (str): Dataset name from Hugging Face.
        use_instruction (bool): Flag to use instruction in prompt.
        use_examples (bool): Flag to use examples in prompt.
        use_post_prompt (bool): Flag to use post_prompt text in prompt.
        tokenizer_path (str): Path to tokenizer (defaults to model_path if not specified).
        wandb_project (str): Weights & Biases project name.
        wandb_entity (str): Weights & Biases entity name.
        wandb_name (str): Weights & Biases run name.
        wandb_dir (str): Weights & Biases output directory.
    """
    if model_path is None:
        model_path = model_name

    # Initialize wandb
    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name or model_name,
            dir=wandb_dir,
            config={
                "model_name": model_name,
                "model_path": model_path,
                "tasks": tasks,
                "lengths": split_names,
                "dataset_name": dataset_name,
                "use_instruction": use_instruction,
                "use_examples": use_examples,
                "use_post_prompt": use_post_prompt,
            },
            job_type="babilong",
        )

    # Load model with vLLM
    print(f"Loading model with vLLM: {model_path}")
    tokenizer = tokenizer_path or model_path
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2000000,
    )
    print("Model loaded successfully with vLLM")

    print(f"prompt template:\n{DEFAULT_TEMPLATE}")

    for task in tqdm(tasks, desc="tasks"):
        # configure the prompt
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"]
            if use_instruction
            else "",
            "examples": DEFAULT_PROMPTS[task]["examples"] if use_examples else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"]
            if use_post_prompt
            else "",
            "template": DEFAULT_TEMPLATE,
        }
        prompt_name = [
            f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
            for k in prompt_cfg
            if k != "template"
        ]
        prompt_name = "_".join(prompt_name)

        for split_name in tqdm(split_names, desc="lengths"):
            # load dataset
            data = datasets.load_dataset(dataset_name, split_name)
            task_data = data[task]

            # Prepare files with predictions, prompt, and generation configurations
            outfile = Path(
                f"{results_folder}/babilong/{task}_{split_name}_{prompt_name}.csv"
            )
            outfile.parent.mkdir(parents=True, exist_ok=True)
            cfg_file = Path(
                f"{results_folder}/babilong/{task}_{split_name}_{prompt_name}.json"
            )
            json.dump(
                {"prompt": prompt_cfg},
                open(cfg_file, "w"),
                indent=4,
            )

            df = pd.DataFrame({"target": [], "output": [], "question": []})

            for sample in tqdm(task_data, desc=f"task: {task} length: {split_name}"):
                target = sample["target"]
                context = sample["input"]
                question = sample["question"]

                # format input text
                input_text = get_formatted_input(
                    context,
                    question,
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )

                # Generate output using vLLM
                output = query_llm(input_text, llm, temperature=0.0, max_tokens=20)
                output = output.strip()

                df.loc[len(df)] = [target, output, question]
                # write results to csv file
                df.to_csv(outfile)

            # Calculate accuracy for this task/length
            # Check if target word appears in the output (case-insensitive)
            accuracy = df.apply(
                lambda row: row["target"].lower() in row["output"].lower(), axis=1
            ).mean()
            if use_wandb:
                wandb.log({
                    f"babilong/{task}/{split_name}/accuracy": accuracy,
                    f"babilong/{task}/{split_name}/total": len(df),
                })

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--results_folder",
        type=str,
        required=True,
        default="./babilong_evals",
        help="Folder to store results",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="RMT-team/babilong",
        help="dataset name from huggingface",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "--model_path", type=str, required=False, help="path to model, optional"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=False, help="path to tokenizer, optional (defaults to model_path)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="List of tasks to evaluate: qa1 qa2 ...",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        required=True,
        help="List of lengths to evaluate: 0k 1k ...",
    )
    parser.add_argument(
        "--use_instruction", action="store_true", help="Use instruction in prompt"
    )
    parser.add_argument(
        "--use_examples", action="store_true", help="Use examples in prompt"
    )
    parser.add_argument(
        "--use_post_prompt", action="store_true", help="Use post prompt text in prompt"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_dir", type=str, default=None, help="Weights & Biases output directory"
    )

    args = parser.parse_args()

    print(args)

    main(
        args.results_folder,
        args.model_name,
        args.model_path,
        args.tasks,
        args.lengths,
        args.dataset_name,
        args.use_instruction,
        args.use_examples,
        args.use_post_prompt,
        args.tokenizer,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_name,
        args.wandb_dir,
    )
