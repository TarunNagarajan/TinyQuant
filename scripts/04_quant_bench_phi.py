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

# --- OPTIMIZED 4-SHOT COT FOR PHI-2 (Shorter, Q&A Format) ---
COT_EXAMPLES = """Q: There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the grove workers plant?
A: Originally 15 trees. After planting there are 21 trees. So 21 - 15 = 6 trees were planted. Answer: 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Started with 3 cars. 2 more arrive. Total: 3 + 2 = 5 cars. Answer: 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32, sister had 42. Total = 32 + 42 = 74. They ate 35. Remaining: 74 - 35 = 39. Answer: 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Started with 20, ended with 12. Gave away: 20 - 12 = 8 lollipops. Answer: 8"""

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
    answer = str(answer).strip().replace(',', '').replace('$', '').replace('%', '').replace(' ', '')
    try:
        num = float(answer)
        if num == int(num): 
            return str(int(num))
        return str(num)
    except:
        return answer

def extract_answer_phi2(text):
    """Phi-2 specific answer extraction."""
    # Try to find explicit "Answer: X" pattern
    answer_match = re.search(r'Answer:\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if answer_match:
        return normalize_numeric_answer(answer_match.group(1))
    
    # Fallback: Look for last number in the generated text
    text_parts = re.split(r'[.!?\n]', text)
    
    # Check last few sentences for numbers
    for part in reversed(text_parts[-3:]):
        nums = re.findall(r'-?\d+(?:\.\d+)?', part)
        if nums:
            return normalize_numeric_answer(nums[-1])
    
    # Ultimate fallback: any number in text
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
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

def visualize_block_structure(model, sensitivity_map, selection_method, percentile):
    """
    Visualizes the block structure and selection for block-aware methods.
    
    INSERT THIS FUNCTION AFTER MODEL LOADING BUT BEFORE EVALUATION.
    """
    if selection_method != "block_aware":
        return  # Skip visualization for non-block methods
    
    print("\n" + "="*80)
    print("BLOCK STRUCTURE VISUALIZATION")
    print("="*80)
    
    from src.selection.block_analysis import BlockAnalyzer
    
    analyzer = BlockAnalyzer(model)
    analyzer.identify_blocks(method="transformer_blocks")
    
    # Get block sensitivities
    block_scores = analyzer.get_block_sensitivity(sensitivity_map)
    sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Determine which blocks will be kept
    num_keep = max(1, int(len(sorted_blocks) * percentile))
    blocks_to_keep_ids = set([block_id for block_id, _ in sorted_blocks[:num_keep]])
    
    print(f"\nKeep Ratio: {percentile*100:.0f}% → Keeping {num_keep}/{len(sorted_blocks)} blocks\n")
    print(f"{'Block':<8} {'Layers':<8} {'Avg Sensitivity':<18} {'Status':<12} {'Visual':<40}")
    print("-"*86)
    
    for block_id, score in sorted_blocks:
        block_layers = analyzer.blocks[block_id]
        kept = block_id in blocks_to_keep_ids
        status = "✓ KEPT FP16" if kept else "✗ QUANTIZED"
        
        # Visual bar
        bar_length = int((score / max(block_scores.values())) * 35)
        bar = "█" * bar_length + "░" * (35 - bar_length)
        
        print(f"{block_id:<8} {len(block_layers):<8} {score:<18.6f} {status:<12} {bar}")
    
    # Show sample layers from kept vs quantized blocks
    print(f"\n{'='*80}")
    print("SAMPLE LAYERS FROM KEPT BLOCKS:")
    print("-"*80)
    for block_id in list(blocks_to_keep_ids)[:3]:  # Show first 3 kept blocks
        print(f"\nBlock {block_id} (Score: {block_scores[block_id]:.6f}):")
        for layer in analyzer.blocks[block_id][:5]:  # Show first 5 layers
            print(f"  - {layer}")
    
    print(f"\n{'='*80}")
    print("SAMPLE LAYERS FROM QUANTIZED BLOCKS:")
    print("-"*80)
    quantized_block_ids = [bid for bid in sorted_blocks if bid[0] not in blocks_to_keep_ids]
    for block_id, score in quantized_block_ids[:3]:  # Show first 3 quantized blocks
        print(f"\nBlock {block_id} (Score: {score:.6f}):")
        for layer in analyzer.blocks[block_id][:5]:
            print(f"  - {layer}")
    
    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "naive", "selective"])
    parser.add_argument("--samples", type=int, default=200)
    
    # Arguments for selective quantization
    parser.add_argument("--selection_method", type=str, default="pct", 
                       choices=["knapsack", "pct", "otsu", "elb", "gradient", "cumulative", "block_aware"])
    parser.add_argument("--sensitivity_method", type=str, default="perturbation", 
                       choices=["perturbation", "fisher", "magnitude"])
    parser.add_argument("--sensitivity_dataset", type=str, default="gsm8k")
    parser.add_argument("--sensitivity_samples", type=int, default=64)
    parser.add_argument("--budget_mb", type=int, default=4096)
    parser.add_argument("--percentile", type=float, default=0.15)
    parser.add_argument("--sensitivity_ratio", type=float, default=0.05)
    parser.add_argument("--budget", type=float, default=0.95)
    
    # Fisher-specific arguments
    parser.add_argument("--fisher_clip_percentile", type=float, default=99.0,
                       help="Percentile for Fisher gradient clipping")
    parser.add_argument("--fisher_clip_samples", type=int, default=32,
                       help="Samples for Fisher clip threshold estimation")
    
    # Block-aware specific
    parser.add_argument("--block_method", type=str, default="transformer_blocks",
                       choices=["transformer_blocks", "weakly_connected", "depth_based"],
                       help="Method for identifying computational blocks")

    args = parser.parse_args()

    Config.set_model("phi")

    # Construct file paths for logging
    if args.mode == "selective":
        log_suffix = f"{args.mode}_{args.selection_method}_{args.sensitivity_method}"
        if args.selection_method == "pct":
            log_suffix += f"_pct{int(args.percentile*100)}"
        elif args.selection_method == "block_aware":
            log_suffix += f"_block{int(args.percentile*100)}"
        elif args.selection_method == "knapsack":
            log_suffix += f"_budget{args.budget_mb}"
    else:
        log_suffix = args.mode

    CHECKPOINT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_checkpoint.jsonl")
    OUTPUT_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}.jsonl")
    SUMMARY_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_summary.json")
    BLOCK_VIZ_FILE = os.path.join(Config.LOGS_DIR, f"gsm8k_{log_suffix}_block_structure.txt")

    print(f"Running Phi-2 Benchmark | Mode: {args.mode}")
    if args.mode == "selective":
        print(f"Selection: {args.selection_method} | Sensitivity: {args.sensitivity_method}")

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
        
        print("[QUANTIZING] Starting selective quantization")
        quantizer = SelectiveQuantizer(model, tokenizer)
        
        # ============================================================
        # CRITICAL INSERTION POINT: VISUALIZE BLOCKS BEFORE QUANTIZING
        # ============================================================
        
        # First compute sensitivity if not already done
        if quantizer.sensitivity_map is None:
            print(f"[COMPUTING SENSITIVITY] Method: {args.sensitivity_method}")
            quantizer.compute_sensitivity(
                args.sensitivity_method,
                args.sensitivity_dataset,
                args.sensitivity_samples,
                fisher_clip_percentile=args.fisher_clip_percentile,
                fisher_clip_samples=args.fisher_clip_samples
            )
        
        # NOW VISUALIZE (only for block_aware method)
        if args.selection_method == "block_aware":
            print("\n[VISUALIZATION] Analyzing block structure...")
            
            # Redirect visualization to file
            import sys
            original_stdout = sys.stdout
            with open(BLOCK_VIZ_FILE, 'w') as f:
                sys.stdout = f
                visualize_block_structure(
                    model, 
                    quantizer.sensitivity_map,
                    args.selection_method,
                    args.percentile
                )
                sys.stdout = original_stdout
            
            # Also print to console
            with open(BLOCK_VIZ_FILE, 'r') as f:
                print(f.read())
            
            print(f"[SAVED] Block structure visualization → {BLOCK_VIZ_FILE}")
        
        # ============================================================
        # NOW PERFORM QUANTIZATION
        # ============================================================
        
        model = quantizer.quantize(
            selection_method=args.selection_method,
            sensitivity_method=args.sensitivity_method,
            dsname=args.sensitivity_dataset,
            n_samples=args.sensitivity_samples,
            budget_mb=args.budget_mb,
            percentile=args.percentile,
            sensitivity_ratio=args.sensitivity_ratio,
            budget=args.budget,
            fisher_clip_percentile=args.fisher_clip_percentile,
            fisher_clip_samples=args.fisher_clip_samples,
            block_method=args.block_method,
            verbose=True
        )

    # CRITICAL FIX: Get actual device AFTER model loading/quantization
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")

    model_size = get_model_size_mb(model)
    print(f"Model Size: {model_size:.2f} MB")

    results = []
    correct = 0
    start_time = time.time()

    print("Starting 4-Shot Chain-of-Thought Evaluation (Phi-2 optimized)...")
    
    for idx in tqdm(range(len(dataset))):
        ex = dataset[idx]
        question = ex['question']
        
        # Phi-2 optimized prompt format (Q&A style, concise)
        prompt_text = f"{COT_EXAMPLES}\n\nQ: {question}\nA:"
        
        # CRITICAL FIX: Use model_device consistently
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the NEW tokens (the generated answer)
        input_len = inputs.input_ids.shape[1]
        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        # Use Phi-2 specific answer extraction
        pred_ans = extract_answer_phi2(response)
        gold_ans = extract_gold_answer(ex["answer"])
        is_correct = (pred_ans == gold_ans) and (pred_ans != "")
        
        if is_correct: 
            correct += 1

        results.append({
            "question": question, 
            "gold": gold_ans, 
            "predicted": pred_ans,
            "response": response,  # Full response for debugging
            "correct": is_correct
        })

        # Checkpoint every 20 samples
        if (idx + 1) % 20 == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                for r in results: 
                    json.dump(r, f)
                    f.write("\n")

    end_time = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    accuracy = correct / len(dataset)

    print(f"\nPhi-2 {args.mode} Result: {accuracy:.2%} | Size: {model_size:.2f}MB | VRAM: {peak_vram:.2f}GB")

    # Save final results (ALL RESPONSES)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"[SAVED] All {len(results)} responses → {OUTPUT_FILE}")

    # Save summary
    summary = {
        "mode": args.mode, 
        "accuracy": accuracy, 
        "size_mb": model_size,
        "peak_vram_gb": peak_vram, 
        "time": end_time - start_time,
        "args": vars(args),
        "correct": correct,
        "total": len(dataset)
    }
    
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[SAVED] Summary statistics → {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
