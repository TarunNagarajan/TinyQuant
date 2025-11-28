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
    # Llama CoT usually ends with "The answer is X."
    # We look for the last number in the text
    text = text.split("Answer:")[-1] # Focus on the generated part
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
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--quant_method", type=str, default="pct", choices=["pct", "otsu", "elb", "gradient", "cumulative"])
    parser.add_argument("--quant_percentile", type=float, default=0.20)
    parser.add_argument("--quant_sensitivity_ratio", type=float, default=0.05)
    parser.add_argument("--quant_budget", type=float, default=0.95)
    args = parser.parse_args()

    Config.set_model("llama_3b")
    
    map_path = os.path.join(Config.MAPS_DIR, args.map_filename)

    CHECKPOINT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{args.mode}_checkpoint.jsonl")
    OUTPUT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{args.mode}.jsonl")
    SUMMARY_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{args.mode}_summary.json")

    print(f"Running Llama 3B Benchmark | Mode: {args.mode}")

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
            load_in_4bit=True,            bnb_4bit_compute_dtype=Config.DTYPE, bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, quantization_config=bnb_config, device_map="auto"
        ).eval()
    elif args.mode == "selective":
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_ID, device_map="auto", torch_dtype=Config.DTYPE
        ).eval()
        with open(map_path, "r") as f: sensitivity_map = json.load(f)
        quantizer = SelectiveQuantizer(model, sensitivity_map)
        model = quantizer.quantize(
            method=args.quant_method,
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
        
        content = f"Solve the following math problems step by step. Show your reasoning clearly.\n\n{COT_EXAMPLES}\n\nQuestion: {question}\nAnswer:"
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        if (idx + 1) % 20 == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                for r in results: json.dump(r, f); f.write("\n")

    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    accuracy = correct / len(dataset)

    print(f"Llama 3B {args.mode} Result: {accuracy:.2%} | Size: {model_size:.2f}MB | VRAM: {peak_vram:.2f}GB")

    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "mode": args.mode, "accuracy": accuracy, "size_mb": model_size,
            "peak_vram_gb": peak_vram, "time": end_time - start_time
        }, f, indent=2)

if __name__ == "__main__":
    main()
