import os
import sys
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.config import Config

def setup_logging(log_path):
    """Setup logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def evaluate_gsm8k(model, tokenizer, device):
    """Evaluate model on GSM8K test set"""
    logging.info("Starting GSM8K evaluation")
    
    # Load GSM8K test dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Take a sample for evaluation (use subset to avoid long evaluation times)
    dataset = dataset.select(range(min(100, len(dataset))))  # Using first 100 samples for efficiency
    
    correct = 0
    total = 0
    
    for i, example in enumerate(tqdm(dataset, desc="GSM8K Evaluation")):
        question = example['question']
        answer = example['answer']
        
        # Format the question for the model
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated answer part
        try:
            answer_part = generated_text.split("Answer:")[-1].strip()
        except:
            answer_part = generated_text
        
        # Check if answer is correct (simple keyword matching for now)
        expected_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        
        # Simple numeric answer check (GSM8K typically has numeric answers)
        try:
            gen_num = float(''.join(filter(lambda x: x.isdigit() or x in '.-', answer_part.split())))
            exp_num = float(''.join(filter(lambda x: x.isdigit() or x in '.-', expected_answer.split())))
            
            if abs(gen_num - exp_num) < 0.01:  # Allow small floating point differences
                correct += 1
        except:
            # If numeric comparison fails, check for exact text match
            if expected_answer.lower() in answer_part.lower():
                correct += 1
        
        total += 1
        
        if i % 10 == 0 and i > 0:  # Log progress every 10 samples
            logging.info(f"GSM8K Progress: {i}/{total}, Accuracy so far: {correct/total:.4f}")
    
    accuracy = correct / total if total > 0 else 0
    logging.info(f"GSM8K Evaluation Complete. Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_wikitext(model, tokenizer, device):
    """Evaluate model on Wikitext using perplexity"""
    logging.info("Starting Wikitext evaluation")
    
    # Load Wikitext validation dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Take a sample for evaluation (use subset to avoid long evaluation times)
    dataset = dataset.filter(lambda x: len(x['text']) > 0 and not x['text'].startswith(" ="))
    dataset = dataset.select(range(min(50, len(dataset))))  # Using first 50 samples for efficiency

    model.eval()
    total_loss = 0
    total_tokens = 0
    
    loss_fn = CrossEntropyLoss(reduction='sum')
    
    for i, example in enumerate(tqdm(dataset, desc="Wikitext Evaluation")):
        text = example['text']
        if not text.strip():
            continue
            
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Add the loss value multiplied by the number of valid tokens
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
        
        if i % 10 == 0 and i > 0:  # Log progress every 10 samples
            current_perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
            logging.info(f"Wikitext Progress: {i}, Current Perplexity: {current_perplexity:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    logging.info(f"Wikitext Evaluation Complete. Perplexity: {perplexity:.4f}")
    return perplexity

def run_baseline_evaluation(eval_type="both", log_dir=None):
    """
    Run baseline evaluation on specified dataset(s)
    
    Args:
        eval_type: "gsm8k", "wikitext", or "both"
        log_dir: Directory to save logs. If None, uses default results/logs
    """
    if log_dir is None:
        log_dir = os.path.join(Config.RESULTS_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
    
    # Load model and tokenizer
    logging.info(f"Loading model: {Config.MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=Config.DEVICE
    )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = Config.DEVICE
    model.to(device)
    
    results = {}
    
    if eval_type in ["gsm8k", "both"]:
        log_path = os.path.join(log_dir, "gsm8k_baseline.log")
        setup_logging(log_path)
        gsm8k_accuracy = evaluate_gsm8k(model, tokenizer, device)
        results["gsm8k"] = gsm8k_accuracy
        
        # Save GSM8K result
        gsm8k_result_file = os.path.join(log_dir, "gsm8k_baseline_result.json")
        with open(gsm8k_result_file, 'w') as f:
            json.dump({"accuracy": gsm8k_accuracy, "dataset": "gsm8k"}, f, indent=2)
    
    if eval_type in ["wikitext", "both"]:
        log_path = os.path.join(log_dir, "wikitext_baseline.log")
        setup_logging(log_path)
        wikitext_perplexity = evaluate_wikitext(model, tokenizer, device)
        results["wikitext"] = wikitext_perplexity
        
        # Save Wikitext result
        wikitext_result_file = os.path.join(log_dir, "wikitext_baseline_result.json")
        with open(wikitext_result_file, 'w') as f:
            json.dump({"perplexity": wikitext_perplexity, "dataset": "wikitext"}, f, indent=2)
    
    # Save combined results
    combined_result_file = os.path.join(log_dir, "baseline_results.json")
    with open(combined_result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Baseline evaluation complete. Results: {results}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", type=str, choices=["gsm8k", "wikitext", "both"], 
                        default="both", help="Type of evaluation to run")
    parser.add_argument("--log_dir", type=str, help="Directory to save logs")
    
    args = parser.parse_args()
    
    run_baseline_evaluation(eval_type=args.eval_type, log_dir=args.log_dir)