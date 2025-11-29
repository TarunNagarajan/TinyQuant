import argparse
import json
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.sensitivity.fisher import compute_fisher, compare_fisher_methods
from src.sensitivity.magnitude import compute_magnitude
from src.sensitivity.perturbation import compute_perturbation_sensitivity
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
        choices=["fisher", "magnitude", "perturbation", "compare_fisher"],
        required=True,
        help="Sensitivity computation method"
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
    parser.add_argument(
        "--fisher_clip_percentile",
        type=float,
        default=99.0,
        help="Percentile for adaptive gradient clipping in Fisher (None to disable). Default: 99.0"
    )
    parser.add_argument(
        "--fisher_clip_samples",
        type=int,
        default=32,
        help="Number of samples for estimating clip threshold. Default: 32"
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
        torch_dtype=Config.DTYPE,
        device_map="auto"
    )

    if args.method == "fisher":
        scores = compute_fisher(
            model, 
            tokenizer, 
            args.dataset, 
            reduction=args.reduction, 
            n_samples=args.n_samples,
            clip_percentile=args.fisher_clip_percentile,
            clip_samples=args.fisher_clip_samples
        )
        
        # Construct filename with clip info
        clip_str = f"clip{int(args.fisher_clip_percentile)}" if args.fisher_clip_percentile else "noclip"
        filename = f"fisher_{args.dataset}_{args.reduction}_{clip_str}.json"
        
    elif args.method == "compare_fisher":
        # Special mode: compare different Fisher settings
        print("\n[COMPARISON MODE] Testing multiple Fisher configurations...")
        results = compare_fisher_methods(
            model, 
            tokenizer, 
            args.dataset, 
            n_samples=args.n_samples
        )
        
        # Save all results
        for method_name, scores in results.items():
            filename = f"fisher_{args.dataset}_{args.reduction}_{method_name}.json"
            output_path = os.path.join(Config.MAPS_DIR, filename)
            print(f"[SAVING] {filename}")
            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2)
        
        print("\n[COMPLETE] All comparison results saved")
        return
        
    elif args.method == "magnitude":
        scores = compute_magnitude(model)
        filename = f"magnitude_{args.dataset}.json"
        
    elif args.method == "perturbation":
        scores = compute_perturbation_sensitivity(
            model, 
            tokenizer, 
            args.dataset, 
            n_samples=args.n_samples
        )
        filename = f"perturbation_{args.dataset}.json"
    else:
        raise ValueError(f"Unknown method: {args.method}")

    output_path = os.path.join(Config.MAPS_DIR, filename)
    print(f"\n[SAVING] {output_path}")

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print("[COMPLETE] Sensitivity computation finished successfully")

if __name__ == "__main__":
    main()
