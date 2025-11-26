import os
import json
import torch
import sys
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import Config
from src.quantization.fisher import compute_fisher
from src.quantization.magnitude import compute_magnitude
from src.data.loader import get_calibration_data

def run_layerwise_sensitivity_analysis():
    print(f"[LOADING MODEL] {Config.MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        device_map=Config.DEVICE
    )

    datasets = ["gsm8k", "wikitext"]
    sample_counts = [64, 128, 256]

    results_summary = []

    print("[STARTING FISHER COMPUTATION]")
    for dataset in datasets:
        for n_samples in sample_counts:
            print(f"[FISHER] [{dataset}] [{n_samples} samples]")
            try:
                original_samples = Config.CALIBRATION_SAMPLES
                Config.CALIBRATION_SAMPLES = n_samples

                scores = compute_fisher(model, tokenizer, dataset)

                Config.CALIBRATION_SAMPLES = original_samples

                filename = f"fisher_{dataset}_{n_samples}.json"
                output_path = os.path.join(Config.MAPS_DIR, filename)

                with open(output_path, "w") as f:
                    json.dump(scores, f, indent=2)

                print(f"[SAVED] {filename}")

                max_fisher = max(scores.values()) if scores else 0
                top_5_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

                results_summary.append({
                    "dataset": dataset,
                    "samples": n_samples,
                    "method": "fisher",
                    "max_fisher": max_fisher,
                    "top_5_layers": top_5_layers
                })

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] Fisher computation failed for {dataset} with {n_samples} samples: {e}")
                continue

    print("[STARTING MAGNITUDE COMPUTATION]")
    for dataset in datasets:
        print(f"[MAGNITUDE] [{dataset}]")
        try:
            scores = compute_magnitude(model)

            filename = f"magnitude_{dataset}.json"
            output_path = os.path.join(Config.MAPS_DIR, filename)

            with open(output_path, "w") as f:
                json.dump(scores, f, indent=2)

            print(f"[SAVED] {filename}")

            max_magnitude = max(scores.values()) if scores else 0
            top_5_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

            results_summary.append({
                "dataset": dataset,
                "samples": None,
                "method": "magnitude",
                "max_magnitude": max_magnitude,
                "top_5_layers": top_5_layers
            })

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Magnitude computation failed for {dataset}: {e}")
            continue

    print("\n[FISHER RESULTS SUMMARY TABLE]")
    print(f"{'Dataset':<12} {'Samples':<8} {'Max Fisher':<12} {'Top 5 Layers':<30}")
    print("-" * 70)

    for result in results_summary:
        if result["method"] == "fisher":
            dataset = result["dataset"]
            samples = result["samples"] if result["samples"] is not None else "N/A"
            max_score = result.get("max_fisher", 0)
            top_5_layers = [layer[0] for layer in result["top_5_layers"]]
            top_5_str = ", ".join(top_5_layers[:3]) + ("..." if len(top_5_layers) > 3 else "")

            print(f"{dataset:<12} {str(samples):<8} {max_score:<12.6f} {top_5_str:<30}")

    print("\n[MAGNITUDE RESULTS SUMMARY]")
    print(f"{'Dataset':<12} {'Max Magnitude':<15} {'Top 5 Layers':<30}")
    print("-" * 60)

    for result in results_summary:
        if result["method"] == "magnitude":
            dataset = result["dataset"]
            max_magnitude = result.get("max_magnitude", 0)
            top_5_layers = [layer[0] for layer in result["top_5_layers"]]
            top_5_str = ", ".join(top_5_layers[:3]) + ("..." if len(top_5_layers) > 3 else "")

            print(f"{dataset:<12} {max_magnitude:<15.6f} {top_5_str:<30}")

    summary_path = os.path.join(Config.MAPS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[SAVED SUMMARY] {summary_path}")
    print("[COMPLETE]")

    return results_summary

if __name__ == "__main__":
    run_layerwise_sensitivity_analysis()