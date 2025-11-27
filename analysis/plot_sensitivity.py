import json
import argparse
import matplotlib.pyplot as plt
import os
import sys

# Ensure the script can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_sensitivity_distribution(sensitivity_map_file):
    """
    Loads a sensitivity map from a JSON file and generates a plot of the
    sensitivity distribution, saving it as a PNG image. It also prints the
    top and bottom 5 most sensitive layers.
    """
    with open(sensitivity_map_file, 'r') as f:
        sensitivity_map = json.load(f)

    # The user provided code to plot and print
    scores = sorted(sensitivity_map.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.yscale('log')
    plt.xlabel('Layer (sorted)')
    plt.ylabel('Fisher Information')
    plt.title('Layer Sensitivity Distribution')
    
    # Save the plot in the same directory as the input file
    output_filename = os.path.splitext(sensitivity_map_file)[0] + '.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
   
    sorted_items = sorted(sensitivity_map.items(), key=lambda x: x[1], reverse=True)
   
    print("\nTop 5 sensitive layers:")
    for name, score in sorted_items[:5]:
        print(f"  {name}: {score:.2e}")
        
    print("\nBottom 5 sensitive layers:")
    for name, score in sorted_items[-5:]:
        print(f"  {name}: {score:.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensitivity distribution from a JSON map.")
    parser.add_argument("sensitivity_map_file", type=str, help="Path to the sensitivity map JSON file.")
    args = parser.parse_args()
    
    plot_sensitivity_distribution(args.sensitivity_map_file)
