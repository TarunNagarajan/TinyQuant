import re
import json
import torch
import sys
import os
import argparse
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantization.quantizer import SelectiveQuantizer
from src.config import Config

COT_EXAMPLES = ""

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
    parser.add_argument("--map_filename", type=str, default="fisher_gsm8k_128.json")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--quant_method", type=str, default="pct", choices=["pct", "otsu", "elb", "gradient", "cumulative"])
    parser.add_argument("--quant_percentile", type=float, default=0.20)
    parser.add_argument("--quant_sensitivity_ratio", type=float, default=0.05)
    parser.add_argument("--quant_budget", type=float, default=0.95)
    args = parser.parse_args()

    # Security: Validate filename to prevent path traversal
    if '..' in args.map_filename or args.map_filename.startswith('/'):
        raise ValueError("Invalid filename provided")

    Config.set_model("qwen_3b")

    map_path = os.path.join(Config.MAPS_DIR, args.map_filename)

    CHECKPOINT_INTERVAL = 50

    # Construct file paths for logging
    if args.mode == "selective":
        log_suffix = f"{args.mode}_{args.quant_method}"
    else:
        log_suffix = args.mode

    CHECKPOINT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_checkpoint.jsonl")
    OUTPUT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}.jsonl")
    SUMMARY_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_summary.json")

    print(f"Running Qwen2.5-3B Benchmark | Mode: {args.mode}")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(args.samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, device_map="auto", torch_dtype=Config.DTYPE
        ).eval()
    elif args.mode == "naive":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=Config.DTYPE, bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, quantization_config=bnb_config, device_map="auto"
        ).eval()
    elif args.mode == "selective":
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, device_map="auto", torch_dtype=Config.DTYPE
        ).eval()
        with open(map_path, "r") as f: sensitivity_map = json.load(f)
        quantizer = SelectiveQuantizer(model, tokenizer)
        # Set the sensitivity map directly
        quantizer.sensitivity_map = sensitivity_map
        model = quantizer.quantize(
            selection_method=args.quant_method,
            percentile=args.quant_percentile,
            sensitivity_ratio=args.quant_sensitivity_ratio,
            budget=args.quant_budget,
            verbose=True
        )

    model_size = get_model_size_mb(model)
    print(f"Model Size: {model_size:.2f} MB")

    results = []
    correct = 0
    start_time = time.time()

    print("Starting 8-Shot Chain-of-Thought Evaluation...")
    
    for idx in tqdm(range(len(dataset))):
        ex = dataset[idx]
        question = ex['question']

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(Config.DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        pred_ans = extract_answer(response)
        gold_ans = extract_gold_answer(ex["answer"])
        is_correct = (pred_ans == gold_ans) and (pred_ans != "")
        if is_correct: correct += 1

        results.append({"question": question, "gold": gold_ans, "predicted": pred_ans, "correct": is_correct})

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                for r in results: json.dump(r, f); f.write("\n")

    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    accuracy = correct / len(dataset)

    print(f"Qwen2.5-3B {args.mode} Result: {accuracy:.2%} | Size: {model_size:.2f}MB | VRAM: {peak_vram:.2f}GB")

    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "mode": args.mode, "accuracy": accuracy, "size_mb": model_size,
            "peak_vram_gb": peak_vram, "time": end_time - start_time
        }, f, indent=2)

if __name__ == "__main__":
    main()
