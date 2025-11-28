import json
import argparse
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Ensure the script can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

def normalize_scores(scores_dict):
    """Normalizes scores in a dictionary to the [0, 1] range."""
    if not scores_dict:
        return {}
    values = np.array(list(scores_dict.values()))
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores_dict}
    
    normalized_scores = {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
    return normalized_scores

def plot_sensitivity_comparison(model_name, metric):
    """
    Loads two sensitivity maps for a given metric (from gsm8k and math datasets),
    generates a comparative plot, and prints top/bottom layers for each.
    """
    Config.set_model(model_name)
    
    # Construct file paths based on convention
    if metric == 'fisher':
        file1 = os.path.join(Config.MAPS_DIR, 'fisher_gsm8k_mean.json')
        file2 = os.path.join(Config.MAPS_DIR, 'fisher_math_mean.json')
        dataset1, dataset2 = 'gsm8k', 'math'
    elif metric == 'magnitude':
        file1 = os.path.join(Config.MAPS_DIR, 'magnitude_gsm8k.json')
        file2 = os.path.join(Config.MAPS_DIR, 'magnitude_math.json')
        dataset1, dataset2 = 'gsm8k', 'math'
    else:
        print(f"Error: Invalid metric '{metric}'. Choose 'fisher' or 'magnitude'.")
        return

    print(f"Loading {file1} and {file2}...")

    try:
        with open(file1, 'r') as f:
            map1 = json.load(f)
        with open(file2, 'r') as f:
            map2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find sensitivity map file. {e}")
        return

    # Normalize scores for plotting
    normalized_map1 = normalize_scores(map1)
    normalized_map2 = normalize_scores(map2)

    # Sort the normalized scores for plotting the distribution
    scores1_sorted = sorted(normalized_map1.values(), reverse=True)
    scores2_sorted = sorted(normalized_map2.values(), reverse=True)
    
    plt.figure(figsize=(12, 7))
    plt.plot(scores1_sorted, label=f'{metric.capitalize()} ({dataset1})', color='blue', alpha=0.8)
    plt.plot(scores2_sorted, label=f'{metric.capitalize()} ({dataset2})', color='red', alpha=0.8)
    
    plt.yscale('log')
    plt.xlabel('Parameter Rank (sorted)')
    plt.ylabel(f'Normalized {metric.capitalize()} Score (log scale)')
    plt.title(f'{metric.capitalize()} Sensitivity Distribution for {model_name}')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Generate a descriptive output filename
    output_dir = Config.MAPS_DIR
    output_filename = os.path.join(output_dir, f'{model_name}_{metric}_{dataset1}_vs_{dataset2}.png')
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
   
    # Print top/bottom 5 for the first map
    sorted_map1 = sorted(map1.items(), key=lambda x: x[1], reverse=True)
    print(f"\n--- {metric.capitalize()} Sensitivity ({dataset1}) ---")
    print("Top 5 sensitive layers:")
    for name, score in sorted_map1[:5]:
        print(f"  {name}: {score:.2e}")
    print("\nBottom 5 sensitive layers:")
    for name, score in sorted_map1[-5:]:
        print(f"  {name}: {score:.2e}")

    # Print top/bottom 5 for the second map
    sorted_map2 = sorted(map2.items(), key=lambda x: x[1], reverse=True)
    print(f"\n--- {metric.capitalize()} Sensitivity ({dataset2}) ---")
    print("Top 5 sensitive layers:")
    for name, score in sorted_map2[:5]:
        print(f"  {name}: {score:.2e}")
    print("\nBottom 5 sensitive layers:")
    for name, score in sorted_map2[-5:]:
        print(f"  {name}: {score:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare sensitivity distributions for gsm8k and math datasets.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to plot (e.g., 'llama_3b').")
    parser.add_argument("--metric", type=str, required=True, choices=['fisher', 'magnitude'], help="The sensitivity metric to plot.")
    args = parser.parse_args()
    
    plot_sensitivity_comparison(args.model_name, args.metric)
