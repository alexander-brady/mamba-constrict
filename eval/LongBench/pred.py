import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def get_pred(data, model, tokenizer, args, fout):
    """Run prediction on data samples."""
    for item in tqdm(data):
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


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

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
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")

    # Load dataset
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

    data = [item for item in data_all if item["_id"] not in has_data]

    # Run prediction
    fout = open(out_file, "a", encoding="utf-8")
    get_pred(data, model, tokenizer, args, fout)
    fout.close()


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
