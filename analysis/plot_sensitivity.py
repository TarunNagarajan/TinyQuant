import json
import argparse
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Ensure the script can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def normalize_scores(scores_dict):
    """Normalizes scores in a dictionary to the [0, 1] range."""
    if not scores_dict:
        return {}
    values = np.array(list(scores_dict.values()))
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores_dict} # Or 0, or 1, depending on desired behavior
    
    normalized_scores = {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
    return normalized_scores

def plot_sensitivity_distribution(fisher_map_file, magnitude_map_file):
    """
    Loads Fisher and Magnitude sensitivity maps from JSON files, generates a
    comparative plot of their normalized distributions, and saves it as a PNG image.
    It also prints the top and bottom 5 most sensitive layers for each metric.
    """
    with open(fisher_map_file, 'r') as f:
        fisher_map = json.load(f)
    with open(magnitude_map_file, 'r') as f:
        magnitude_map = json.load(f)

    # Normalize scores for plotting
    normalized_fisher = normalize_scores(fisher_map)
    normalized_magnitude = normalize_scores(magnitude_map)

    # Sort the normalized scores for plotting the distribution
    fisher_scores_sorted = sorted(normalized_fisher.values(), reverse=True)
    magnitude_scores_sorted = sorted(normalized_magnitude.values(), reverse=True)
    
    plt.figure(figsize=(12, 7))
    plt.plot(fisher_scores_sorted, label='Fisher (Normalized)', color='blue', alpha=0.8)
    plt.plot(magnitude_scores_sorted, label='Magnitude (Normalized)', color='red', alpha=0.8)
    
    plt.yscale('log')
    plt.xlabel('Parameter Rank (sorted)')
    plt.ylabel('Normalized Score (log scale)')
    plt.title('Fisher vs. Magnitude Sensitivity Distribution')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Generate a descriptive output filename
    base_name = os.path.basename(fisher_map_file).replace('fisher_', '').replace('.json', '')
    output_filename = os.path.join(os.path.dirname(fisher_map_file), f'{base_name}_fisher_vs_magnitude.png')
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
   
    # Print top/bottom 5 for original Fisher scores
    sorted_fisher = sorted(fisher_map.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Fisher Sensitivity ---")
    print("Top 5 sensitive layers:")
    for name, score in sorted_fisher[:5]:
        print(f"  {name}: {score:.2e}")
    print("\nBottom 5 sensitive layers:")
    for name, score in sorted_fisher[-5:]:
        print(f"  {name}: {score:.2e}")

    # Print top/bottom 5 for original Magnitude scores
    sorted_magnitude = sorted(magnitude_map.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Magnitude Sensitivity ---")
    print("Top 5 sensitive layers:")
    for name, score in sorted_magnitude[:5]:
        print(f"  {name}: {score:.2e}")
    print("\nBottom 5 sensitive layers:")
    for name, score in sorted_magnitude[-5:]:
        print(f"  {name}: {score:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare Fisher and Magnitude sensitivity distributions from JSON maps.")
    parser.add_argument("fisher_map_file", type=str, help="Path to the Fisher sensitivity map JSON file.")
    parser.add_argument("magnitude_map_file", type=str, help="Path to the Magnitude sensitivity map JSON file.")
    args = parser.parse_args()
    
    plot_sensitivity_distribution(args.fisher_map_file, args.magnitude_map_file)
