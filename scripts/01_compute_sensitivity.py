import argparse
import json
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantization.fisher import compute_fisher
from src.quantization.magnitude import compute_magnitude
from src.config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["qwen", "qwen_3b", "phi", "stablelm", "llama_3b"],
        help="Name of the model to use"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fisher", "magnitude"],
        required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "wikitext", "math"],
        default="gsm8k"
    )
    parser.add_argument(
        "--reduction",
        type=str,
        choices=["mean", "sum"],
        default="mean",
        help="Fisher reduction mode"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=Config.CALIBRATION_SAMPLES,
        help="Number of samples to use for calibration"
    )

    args = parser.parse_args()

    # Set the model configuration
    Config.set_model(args.model_name)

    print(f"[LOADING MODEL] [{Config.MODEL_ID}]")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_ID,
        dtype=Config.DTYPE,
        device_map=Config.DEVICE
    )

    if args.method == "fisher":
        scores = compute_fisher(model, tokenizer, args.dataset, reduction=args.reduction, n_samples=args.n_samples)
        filename = f"fisher_{args.dataset}_{args.reduction}.json"
    else:
        scores = compute_magnitude(model)
        filename = f"magnitude_{args.dataset}.json"

    output_path = os.path.join(Config.MAPS_DIR, filename)
    print(f"[SAVING]")

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("[COMPLETE]")

if __name__ == "__main__":
    main()
