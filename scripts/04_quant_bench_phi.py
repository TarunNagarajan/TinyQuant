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

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantization.quantizer import SelectiveQuantizer
from src.config import Config

# --- 8-SHOT COT EXAMPLES ---
COT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after planting. So the number of trees planted is 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are 3 cars already, and 2 more arrive, for a total of 3 + 2 = 5 cars. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Leah had 32 chocolates and her sister had 42. In total they had 32 + 42 = 74. They ate 35. So they have 74 - 35 = 39 pieces left. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20. He ended with 12. He gave away 20 - 12 = 8 lollipops. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn has 5 toys. Mom gave him 2. Dad gave him 2. Total new toys = 2 + 2 = 4. Total toys now = 5 + 4 = 9. The answer is 9.

Question: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room?
Answer: There were 9 computers. 5 computers were added for 4 days. So 5 * 4 = 20 computers were added. Total computers = 9 + 20 = 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58. Tuesday he lost 23, so he had 58 - 23 = 35. Wednesday he lost 2 more, so 35 - 2 = 33. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: She bought 5 bagels. Each cost $3. Total cost = 5 * 3 = 15. She started with $23. Money left = 23 - 15 = 8. The answer is 8."""

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def normalize_numeric_answer(answer):
    if not answer: return ""
    answer = str(answer).strip().replace(',', '').replace('$', '').replace('%', '').replace(' ', '')
    try:
        num = float(answer)
        if num == int(num): return str(int(num))
        return str(num)
    except:
        return answer

def extract_answer(text):
    # Phi-2 generates raw completion, extract last number
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
    parser.add_argument("--samples", type=int, default=200)
    # Arguments for selective quantization
    parser.add_argument("--selection_method", type=str, default="pct", choices=["knapsack", "pct", "otsu", "elb", "gradient", "cumulative"])
    parser.add_argument("--sensitivity_method", type=str, default="perturbation", choices=["perturbation", "fisher", "magnitude"])
    parser.add_argument("--sensitivity_dataset", type=str, default="gsm8k")
    parser.add_argument("--sensitivity_samples", type=int, default=64)
    parser.add_argument("--budget_mb", type=int, default=4096)
    parser.add_argument("--percentile", type=float, default=0.15)
    parser.add_argument("--sensitivity_ratio", type=float, default=0.05)
    parser.add_argument("--budget", type=float, default=0.95)

    args = parser.parse_args()

    Config.set_model("phi")

    # Construct file paths for logging
    if args.mode == "selective":
        log_suffix = f"{args.mode}_{args.selection_method}_{args.sensitivity_method}"
        if args.selection_method == "pct":
            log_suffix += f"_pct{int(args.percentile*100)}"
        elif args.selection_method == "knapsack":
            log_suffix += f"_budget{args.budget_mb}"
    else:
        log_suffix = args.mode

    CHECKPOINT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_checkpoint.jsonl")
    OUTPUT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}.jsonl")
    SUMMARY_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_summary.json")

    print(f"Running Phi-2 Benchmark | Mode: {args.mode}")

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(args.samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "baseline":
        print("[LOADING] Baseline FP16 model")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).eval()
        
    elif args.mode == "naive":
        print("[LOADING] Naive INT4 quantized model")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, 
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        
    elif args.mode == "selective":
        print("[LOADING] Model for selective quantization")
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).eval()
        
        # CRITICAL: No dtype conversion after loading
        print("[QUANTIZING] Starting selective quantization")
        quantizer = SelectiveQuantizer(model, tokenizer)
        model = quantizer.quantize(
            selection_method=args.selection_method,
            sensitivity_method=args.sensitivity_method,
            dsname=args.sensitivity_dataset,
            n_samples=args.sensitivity_samples,
            budget_mb=args.budget_mb,
            percentile=args.percentile,
            sensitivity_ratio=args.sensitivity_ratio,
            budget=args.budget,
            verbose=True
        )

    # CRITICAL: Get actual device after quantization
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")

    model_size = get_model_size_mb(model)
    print(f"Model Size: {model_size:.2f} MB")

    results = []
    correct = 0
    start_time = time.time()

    print("Starting 8-Shot Chain-of-Thought Evaluation...")
    
    for idx in tqdm(range(len(dataset))):
        ex = dataset[idx]
        question = ex['question']
        
        # Construct Prompt for Phi-2 (Raw completion format)
        prompt_text = f"Solve the following math problems step by step. Show your reasoning clearly.\n\n{COT_EXAMPLES}\n\nQuestion: {question}\nAnswer:"
        
        # CRITICAL: Send to model_device, not Config.DEVICE
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the NEW tokens
        input_len = inputs.input_ids.shape[1]
        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        pred_ans = extract_answer(response)
        gold_ans = extract_gold_answer(ex["answer"])
        is_correct = (pred_ans == gold_ans) and (pred_ans != "")
        if is_correct: 
            correct += 1

        results.append({
            "question": question, 
            "gold": gold_ans, 
            "predicted": pred_ans, 
            "correct": is_correct
        })

        if (idx + 1) % 20 == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                for r in results: 
                    json.dump(r, f)
                    f.write("\n")

    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    accuracy = correct / len(dataset)

    print(f"\nPhi-2 {args.mode} Result: {accuracy:.2%} | Size: {model_size:.2f}MB | VRAM: {peak_vram:.2f}GB")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "mode": args.mode, 
            "accuracy": accuracy, 
            "size_mb": model_size,
            "peak_vram_gb": peak_vram, 
            "time": end_time - start_time,
            "args": vars(args)
        }, f, indent=2)

if __name__ == "__main__":
    main()
