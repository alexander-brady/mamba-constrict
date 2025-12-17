import argparse
import logging
import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

template_rag = open("prompts/0shot_rag.txt", encoding="utf-8").read()
template_no_context = open("prompts/0shot_no_context.txt", encoding="utf-8").read()
template_0shot = open("prompts/0shot.txt", encoding="utf-8").read()
template_0shot_cot = open("prompts/0shot_cot.txt", encoding="utf-8").read()
template_0shot_cot_ans = open("prompts/0shot_cot_ans.txt", encoding="utf-8").read()


def query_llm(
    prompt,
    model,
    tokenizer,
    temperature=0.5,
    max_new_tokens=128,
):
    """Query the local model with the given prompt."""
    # Tokenize prompt (no truncation for SSMs)
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Generate
    input_tensor = torch.tensor([input_ids]).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = output_ids[0][len(input_ids):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(data, model, tokenizer, args, fout, model_name):
    """Run prediction on data samples."""
    correct_count = 0
    total_processed = 0

    for item in tqdm(data, desc=f"[{model_name}] Processing"):
        context = item["context"]
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][: args.rag]
            retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
            context = "\n\n".join(
                [
                    f"Retrieved chunk {idx + 1}: {x['content']}"
                    for idx, x in enumerate(retrieved)
                ]
            )
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = (
            template.replace("$DOC$", context.strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip())
        )
        if args.cot:
            output = query_llm(
                prompt, model, tokenizer, temperature=0.1, max_new_tokens=1024
            )
        else:
            output = query_llm(
                prompt, model, tokenizer, temperature=0.1, max_new_tokens=128
            )
        if output == "":
            continue
        if args.cot:  # extract answer
            response = output.strip()
            item["response_cot"] = response
            prompt = (
                template_0shot_cot_ans.replace("$DOC$", context.strip())
                .replace("$Q$", item["question"].strip())
                .replace("$C_A$", item["choice_A"].strip())
                .replace("$C_B$", item["choice_B"].strip())
                .replace("$C_C$", item["choice_C"].strip())
                .replace("$C_D$", item["choice_D"].strip())
                .replace("$COT$", response)
            )
            output = query_llm(
                prompt, model, tokenizer, temperature=0.1, max_new_tokens=128
            )
            if output == "":
                continue
        response = output.strip()
        item["response"] = response
        item["pred"] = extract_answer(response)
        item["judge"] = item["pred"] == item["answer"]
        item["context"] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.flush()

        total_processed += 1
        if item["judge"]:
            correct_count += 1

        # Log progress every 10 samples
        if total_processed % 10 == 0:
            accuracy = (correct_count / total_processed) * 100
            logger.info(
                f"[{model_name}] Progress: {total_processed}/{len(data)} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    # Final summary
    if total_processed > 0:
        final_accuracy = (correct_count / total_processed) * 100
        logger.info(
            f"[{model_name}] Final: {total_processed} samples | "
            f"Accuracy: {final_accuracy:.2f}% ({correct_count}/{total_processed})"
        )


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    logger.info("=" * 60)
    logger.info(f"LongBench-v2 Evaluation - Model: {model_name}")
    logger.info("=" * 60)

    # Determine output file
    if args.rag > 0:
        out_file = os.path.join(
            args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl"
        )
    elif args.no_context:
        out_file = os.path.join(
            args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl"
        )
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    # Load model and tokenizer
    logger.info(f"[{model_name}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"[{model_name}] Model loaded on device: {model.device}")

    # Load dataset
    logger.info(f"[{model_name}] Loading dataset...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    data_all = [
        {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        for item in dataset
    ]

    # Skip already processed samples
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
        logger.info(f"[{model_name}] Resuming: {len(has_data)} already processed")

    data = [item for item in data_all if item["_id"] not in has_data]
    logger.info(f"[{model_name}] Processing {len(data)}/{len(data_all)} samples")

    # Run prediction
    fout = open(out_file, "a", encoding="utf-8")
    get_pred(data, model, tokenizer, args, fout, model_name)
    fout.close()
    logger.info("=" * 60)
    logger.info(f"LongBench-v2 Completed - Model: {model_name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path (HuggingFace or local)")
    parser.add_argument(
        "--cot", "-cot", action="store_true"
    )  # set to True if using COT
    parser.add_argument(
        "--no_context", "-nc", action="store_true"
    )  # set to True if using no context (directly measuring memorization)
    parser.add_argument(
        "--rag", "-rag", type=int, default=0
    )  # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    args = parser.parse_args()
    main()
