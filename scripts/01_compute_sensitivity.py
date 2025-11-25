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
    parser.add_argument("--method", type = str, choices = ["fisher", "magnitude"], required = True)
    parser.add_argument("--dataset", type = str, choices = ["gsm8k", "wikitext"], default = "gsm8k")
    args = parser.parse_args()

    print(f"[LOADING MODEL] [{Config.MODEL_ID}]")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_ID, dtype = Config.DTYPE, device_map = Config.DEVICE)

    if args.method == "fisher":
        scores = compute_fisher(model, tokenizer, args.dataset)
        filename = f"fisher_{args.dataset}.json"
    else:
        scores = compute_magnitude(model)
        filename = f"magnitude_{args.dataset}.json"

    output_path = os.path.join(Config.MAPS_DIR, filename)
    print(f"[SAVING]")

    with open(output_path, "w") as f:
        json.dump(scores, f)

    print("[COMPLETE]")

if __name__ == "__main__":
    main()


