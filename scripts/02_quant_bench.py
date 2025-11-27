import re
import json
import torch
import gc
import sys
import os
import argparse
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantization.quantizer import SelectiveQuantizer

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def normalize_numeric_answer(answer):
    if not answer:
        return ""
    answer = str(answer).strip()
    answer = answer.replace(',', '')
    answer = answer.replace('$', '')
    answer = answer.replace('%', '')
    answer = answer.replace(' ', '')
    try:
        num = float(answer)
        if not (-1e15 < num < 1e15):
            return answer
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer

def extract_answer(text):
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return normalize_numeric_answer(boxed_match.group(1))
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return normalize_numeric_answer(nums[-1])
    return ""

def extract_gold_answer(answer_text):
    parts = answer_text.strip().split("####")
    if len(parts) >= 2:
        gold = parts[-1].strip()
    else:
        nums = re.findall(r"-?\d+\.?\d*", answer_text)
        gold = nums[-1] if nums else ""
    return normalize_numeric_answer(gold)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "naive", "selective"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--map_path", type=str, default="results/maps/fisher_gsm8k_128.json")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_INTERVAL = 50
    CHECKPOINT_FILE = f"gsm8k_{args.mode}_checkpoint.jsonl"
    OUTPUT_FILE = f"gsm8k_{args.mode}.jsonl"
    SUMMARY_FILE = f"gsm8k_{args.mode}_summary.json"

    print(f"Mode: {args.mode}")
    print(f"Device: {DEVICE}")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(args.samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.mode == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()

    elif args.mode == "naive":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto"
        ).eval()

    elif args.mode == "selective":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        
        with open(args.map_path, "r") as f:
            sensitivity_map = json.load(f)
        
        quantizer = SelectiveQuantizer(model, sensitivity_map)
        model = quantizer.quantize(method="kmeans", verbose=True)

    model_size = get_model_size_mb(model)
    print(f"Model Size: {model_size:.2f} MB")

    start_idx = 0
    results = []
    correct = 0

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from {CHECKPOINT_FILE}")
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                results.append(result)
                if result["correct"]:
                    correct += 1
        start_idx = len(results)

    start_time = time.time()

    for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx, total=len(dataset)):
        ex = dataset[idx]
        prompt = ex['question']

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        pred_ans = extract_answer(response)
        gold_ans = extract_gold_answer(ex["answer"])

        is_correct = (pred_ans == gold_ans) and (pred_ans != "")
        if is_correct:
            correct += 1

        results.append({
            "question": prompt,
            "gold": gold_ans,
            "predicted": pred_ans,
            "correct": is_correct
        })

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                for r in results:
                    json.dump(r, f, ensure_ascii=False)
                    f.write("\n")

    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    accuracy = correct / len(dataset)
    
    print(f"Results for {args.mode}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Size: {model_size:.2f} MB")
    print(f"Peak VRAM: {peak_vram:.2f} GB")

    summary = {
        "mode": args.mode,
        "model": args.model,
        "accuracy": accuracy,
        "size_mb": model_size,
        "peak_vram_gb": peak_vram,
        "total_time_sec": end_time - start_time
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
