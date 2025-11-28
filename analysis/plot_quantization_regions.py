import json
import argparse
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Ensure the script can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.quantization.quantizer import SelectiveQuantizer

def plot_quantization_regions(model_name):
    sensitivity_map_file = os.path.join(Config.MAPS_DIR, 'fisher_gsm8k_mean.json')
    print(f"Generating quantization region plot for {model_name} using {sensitivity_map_file}...")

    try:
        with open(sensitivity_map_file, 'r') as f:
            sensitivity_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Default sensitivity map file not found at {sensitivity_map_file}")
        print("Please ensure you have run the sensitivity analysis with '--method fisher' and '--dataset gsm8k'.")
        return

    # --- Calculate Thresholds ---
    pct_threshold = SelectiveQuantizer.get_threshold_pct(sensitivity_map, percentile=0.2)
    elb_threshold = SelectiveQuantizer.get_threshold_elb(sensitivity_map)
    grad_threshold = SelectiveQuantizer.find_optimal_threshold(sensitivity_map, sensitivity_ratio=0.05)
    cum_threshold = SelectiveQuantizer.cumulative_budget_threshold(sensitivity_map, budget=0.95)

    thresholds = {
        "20th Percentile": pct_threshold,
        "Elbow Method": elb_threshold,
        "Gradient Method (ratio=0.05)": grad_threshold,
        "Cumulative Budget (budget=0.95)": cum_threshold,
    }

    # --- Plotting ---
    scores = sorted(sensitivity_map.values(), reverse=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'Quantization Regions for {Config.MODEL_ID}\n(Based on {os.path.basename(sensitivity_map_file)})', fontsize=16)

    for i, (method_name, threshold) in enumerate(thresholds.items()):
        ax = axes[i // 2, i % 2]
        
        # Find the rank by counting how many scores are above the threshold
        threshold_rank = np.sum(np.array(scores) > threshold)

        ax.plot(scores, color='navy')
        
        ax.axvspan(0, threshold_rank, color=(1, 0, 0, 0.2), label=f'High Sensitivity (Keep FP16/BF16)')
        ax.axvspan(threshold_rank, len(scores), color=(0, 1, 0, 0.2), label=f'Low Sensitivity (Quantize to INT4)')
        ax.axvline(x=threshold_rank, color='black', linestyle='--', label=f'Threshold Rank: {threshold_rank} (Score: {threshold:.2e})')
        
        ax.set_yscale('log')
        ax.set_title(f'Method: {method_name}')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # Set shared labels
    for ax in axes[:, 0]:
        ax.set_ylabel('Sensitivity Score (log scale)')
    for ax in axes[1, :]:
        ax.set_xlabel('Parameter Rank (sorted by sensitivity)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    output_dir = os.path.join(Config.RESULTS_DIR, 'reports', model_name)
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(sensitivity_map_file).replace('.json', '')
    output_filename = os.path.join(output_dir, f'quantization_regions_{base_filename}.png')
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot quantization regions based on the default Fisher sensitivity map.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for which to plot quantization regions.")
    
    args = parser.parse_args()

    # Set model config to ensure paths are correct
    Config.set_model(args.model_name)
    
    plot_quantization_regions(args.model_name)
