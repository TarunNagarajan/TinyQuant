import os
import sys
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import Config

def setup_logging(log_path):
    """Setup logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def evaluate_gsm8k(model, tokenizer, device, n_samples=100):
    """Evaluate model on GSM8K test set"""
    logging.info("Starting GSM8K evaluation")
    
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for i, example in enumerate(tqdm(dataset, desc="GSM8K Evaluation")):
        question = example['question']
        answer = example['answer']
        
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
        
        try:
            answer_part = generated_text.split("Answer:")[-1].strip()
        except:
            answer_part = generated_text
        
        expected_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        
        try:
            gen_num = float(''.join(filter(lambda x: x.isdigit() or x in '.-', answer_part.split())))
            exp_num = float(''.join(filter(lambda x: x.isdigit() or x in '.-', expected_answer.split())))
            
            if abs(gen_num - exp_num) < 0.01:
                correct += 1
        except:
            if expected_answer.lower() in answer_part.lower():
                correct += 1
        
        total += 1
        
        if i % 10 == 0 and i > 0:
            logging.info(f"GSM8K Progress: {i}/{len(dataset)}, Accuracy so far: {correct/total:.4f}")
    
    accuracy = correct / total if total > 0 else 0
    logging.info(f"GSM8K Evaluation Complete. Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_wikitext(model, tokenizer, device, n_samples=50):
    """Evaluate model on Wikitext using perplexity"""
    logging.info("Starting Wikitext evaluation")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    dataset = dataset.filter(lambda x: len(x['text']) > 0 and not x['text'].startswith(" ="))
    dataset = dataset.select(range(min(n_samples, len(dataset))))

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
            
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
        
        if i % 10 == 0 and i > 0:
            current_perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
            logging.info(f"Wikitext Progress: {i}/{len(dataset)}, Current Perplexity: {current_perplexity:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    logging.info(f"Wikitext Evaluation Complete. Perplexity: {perplexity:.4f}")
    return perplexity

def run_baseline_evaluation(model_name, eval_type="both", gsm8k_samples=100, wikitext_samples=50):
    Config.set_model(model_name)
    
    log_path = os.path.join(Config.LOGS_DIR, "baseline_eval.log")
    setup_logging(log_path)
    
    logging.info(f"Loading model: {Config.MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=Config.DEVICE
    )
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = Config.DEVICE
    model.to(device)
    
    results = {}
    
    if eval_type in ["gsm8k", "both"]:
        gsm8k_accuracy = evaluate_gsm8k(model, tokenizer, device, n_samples=gsm8k_samples)
        results["gsm8k_accuracy"] = gsm8k_accuracy
    
    if eval_type in ["wikitext", "both"]:
        wikitext_perplexity = evaluate_wikitext(model, tokenizer, device, n_samples=wikitext_samples)
        results["wikitext_perplexity"] = wikitext_perplexity
    
    combined_result_file = os.path.join(Config.LOGS_DIR, "baseline_results.json")
    with open(combined_result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Baseline evaluation complete. Results: {results}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["qwen", "qwen_3b", "phi"], help="Name of the model to evaluate")
    parser.add_argument("--eval_type", type=str, choices=["gsm8k", "wikitext", "both"], 
                        default="both", help="Type of evaluation to run")
    parser.add_argument("--gsm8k_samples", type=int, default=100, help="Number of samples for GSM8K evaluation")
    parser.add_argument("--wikitext_samples", type=int, default=50, help="Number of samples for Wikitext evaluation")
    
    args = parser.parse_args()
    
    run_baseline_evaluation(
        model_name=args.model_name,
        eval_type=args.eval_type,
        gsm8k_samples=args.gsm8k_samples,
        wikitext_samples=args.wikitext_samples
    )