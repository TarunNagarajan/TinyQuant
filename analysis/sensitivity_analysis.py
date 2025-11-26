import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import Config

def organize_and_preprocess_data():
    """
    Load all Fisher and magnitude JSON results and convert to structured DataFrame
    """
    maps_dir = Config.MAPS_DIR
    
    # Find all JSON files in the maps directory
    json_files = [f for f in os.listdir(maps_dir) if f.endswith('.json') and f != 'summary.json']
    
    data_dict = {}
    
    for file in json_files:
        file_path = os.path.join(maps_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        data_dict[file] = data
    
    return data_dict

def aggregate_fisher_per_layer(fisher_data):
    """
    Aggregate Fisher values per layer by summing or averaging weights in each layer
    """
    layer_aggregated = defaultdict(lambda: {"fisher_sum": 0, "fisher_mean": 0, "count": 0})
    
    for file_name, data in fisher_data.items():
        if file_name.startswith('fisher_'):
            for param_name, fisher_value in data.items():
                # Extract layer name by splitting on common PyTorch naming patterns
                if '.weight' in param_name:
                    layer_name = param_name.rsplit('.', 1)[0]  # Remove '.weight'
                else:
                    layer_name = param_name
                
                layer_aggregated[layer_name]["fisher_sum"] += fisher_value
                layer_aggregated[layer_name]["count"] += 1
    
    # Calculate mean Fisher value per layer
    for layer_name in layer_aggregated:
        count = layer_aggregated[layer_name]["count"]
        if count > 0:
            layer_aggregated[layer_name]["fisher_mean"] = layer_aggregated[layer_name]["fisher_sum"] / count
    
    return dict(layer_aggregated)

def identify_top_k_sensitive_layers(layer_data, k=5):
    """
    Identify top K sensitive layers based on Fisher values
    """
    # Sort by Fisher mean (descending)
    sorted_layers = sorted(layer_data.items(), key=lambda x: x[1]["fisher_mean"], reverse=True)
    return sorted_layers[:k]

def group_by_module_type(layer_data):
    """
    Group parameters by module type (attention, MLP, layernorm, embedding)
    """
    module_groups = defaultdict(list)
    
    for layer_name, data in layer_data.items():
        # Classify based on layer name patterns
        if 'attn' in layer_name or 'self_attn' in layer_name:
            module_type = 'attention'
        elif 'mlp' in layer_name or 'ffn' in layer_name or 'down_proj' in layer_name or 'up_proj' in layer_name or 'gate_proj' in layer_name:
            module_type = 'mlp'
        elif 'norm' in layer_name:
            module_type = 'layernorm'
        elif 'embed' in layer_name or 'wte' in layer_name or 'wpe' in layer_name:
            module_type = 'embedding'
        elif 'q_proj' in layer_name or 'k_proj' in layer_name or 'v_proj' in layer_name:
            module_type = 'attention_qkv'
        elif 'o_proj' in layer_name:
            module_type = 'attention_o'
        else:
            module_type = 'other'
        
        module_groups[module_type].append((layer_name, data))
    
    return dict(module_groups)

def compute_module_statistics(module_groups):
    """
    Compute statistics per module type
    """
    module_stats = {}
    
    for module_type, layers in module_groups.items():
        fisher_means = [layer[1]["fisher_mean"] for layer in layers]
        fisher_sums = [layer[1]["fisher_sum"] for layer in layers]
        
        module_stats[module_type] = {
            "mean_fisher_mean": np.mean(fisher_means) if fisher_means else 0,
            "max_fisher_mean": np.max(fisher_means) if fisher_means else 0,
            "min_fisher_mean": np.min(fisher_means) if fisher_means else 0,
            "std_fisher_mean": np.std(fisher_means) if fisher_means else 0,
            "count": len(layers)
        }
    
    return module_stats

def magnitude_vs_sensitivity_analysis(fisher_data, magnitude_data):
    """
    Cross-analysis between magnitude and sensitivity
    """
    # Find common parameters between Fisher and magnitude data
    result = []
    
    for fisher_file, fisher_values in fisher_data.items():
        if fisher_file.startswith('fisher_'):
            # Find corresponding magnitude file
            dataset = fisher_file.split('_')[1]  # gsm8k or wikitext
            magnitude_file = f"magnitude_{dataset}.json"
            
            if magnitude_file in magnitude_data:
                mag_values = magnitude_data[magnitude_file]
                
                # Find common parameters
                common_params = set(fisher_values.keys()) & set(mag_values.keys())
                
                for param in common_params:
                    result.append({
                        'param_name': param,
                        'fisher_value': fisher_values[param],
                        'magnitude_value': mag_values[param],
                        'dataset': dataset
                    })
    
    return result

def create_parameter_ranking(fisher_data):
    """
    Create a ranked list of all parameters by Fisher value
    """
    all_params = []
    
    for file_name, data in fisher_data.items():
        if file_name.startswith('fisher_'):
            dataset = file_name.split('_')[1]  # gsm8k or wikitext
            for param_name, fisher_value in data.items():
                all_params.append({
                    'param_name': param_name,
                    'fisher_value': fisher_value,
                    'dataset': dataset
                })
    
    # Sort by Fisher value (descending)
    all_params.sort(key=lambda x: x['fisher_value'], reverse=True)
    
    return all_params

def generate_quantization_zones(param_ranking, top_percent=0.1):
    """
    Generate quantization zones based on parameter rankings
    """
    n_params = len(param_ranking)
    zone_a_count = int(n_params * top_percent)  # Top 10% most sensitive
    
    for i, param in enumerate(param_ranking):
        if i < zone_a_count:
            param['zone'] = 'A'  # Most sensitive - avoid quantization
        elif i < zone_a_count * 2:
            param['zone'] = 'B'  # Moderately sensitive - consider 4-8 bit
        else:
            param['zone'] = 'C'  # Low sensitivity - aggressive quantization possible
    
    return param_ranking

def generate_reports(aggregated_data, module_stats, param_rankings, output_dir):
    """
    Generate comprehensive reports
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Top sensitive layers report
    top_layers = identify_top_k_sensitive_layers(aggregated_data, k=10)
    top_layers_df = pd.DataFrame([
        {
            'layer_name': layer[0],
            'fisher_mean': layer[1]['fisher_mean'],
            'fisher_sum': layer[1]['fisher_sum'],
            'parameter_count': layer[1]['count']
        }
        for layer in top_layers
    ])
    
    # 2. Module statistics report
    module_stats_df = pd.DataFrame.from_dict(module_stats, orient='index')
    module_stats_df.index.name = 'module_type'
    
    # 3. Parameter ranking report
    param_rankings_df = pd.DataFrame(param_rankings)
    
    # Save reports
    top_layers_df.to_csv(os.path.join(output_dir, 'top_sensitive_layers.csv'), index=False)
    module_stats_df.to_csv(os.path.join(output_dir, 'module_statistics.csv'))
    param_rankings_df.to_csv(os.path.join(output_dir, 'parameter_rankings.csv'), index=False)
    
    # 4. Summary text file
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Top 10 Most Sensitive Layers:\n")
        for i, layer in enumerate(top_layers[:10]):
            f.write(f"{i+1:2d}. {layer[0]}: Fisher Mean = {layer[1]['fisher_mean']:.6f}\n")
        
        f.write(f"\nModule Type Statistics:\n")
        for module_type, stats in module_stats.items():
            f.write(f"{module_type}: Mean Fisher = {stats['mean_fisher_mean']:.6f}, "
                    f"Max Fisher = {stats['max_fisher_mean']:.6f}, Count = {stats['count']}\n")
    
    return top_layers_df, module_stats_df, param_rankings_df

def run_full_analysis():
    """
    Run the complete sensitivity analysis as outlined
    """
    print("[LOADING] All Fisher and magnitude JSON results")
    all_data = organize_and_preprocess_data()
    
    # Separate Fisher and magnitude data
    fisher_data = {k: v for k, v in all_data.items() if k.startswith('fisher_')}
    magnitude_data = {k: v for k, v in all_data.items() if k.startswith('magnitude_')}
    
    print("[AGGREGATING] Fisher values per layer")
    layer_aggregated = aggregate_fisher_per_layer(fisher_data)
    
    print("[GROUPING] Parameters by module type")
    module_groups = group_by_module_type(layer_aggregated)
    module_stats = compute_module_statistics(module_groups)
    
    print("[ANALYZING] Magnitude vs Sensitivity")
    mag_vs_sens = magnitude_vs_sensitivity_analysis(fisher_data, magnitude_data)
    
    print("[RANKING] Parameters by sensitivity")
    param_rankings = create_parameter_ranking(fisher_data)
    param_rankings_with_zones = generate_quantization_zones(param_rankings)
    
    print("[GENERATING] Reports")
    output_dir = os.path.join(Config.RESULTS_DIR, 'reports')
    top_layers_df, module_stats_df, param_rankings_df = generate_reports(
        layer_aggregated, module_stats, param_rankings_with_zones, output_dir
    )

    print(f"[SAVED] Analysis reports to {output_dir}")
    print("[COMPLETE] Sensitivity analysis finished")
    
    return {
        'top_layers': top_layers_df,
        'module_stats': module_stats_df,
        'param_rankings': param_rankings_df,
        'layer_aggregated': layer_aggregated,
        'module_groups': module_groups,
        'magnitude_vs_sensitivity': mag_vs_sens
    }

if __name__ == "__main__":
    results = run_full_analysis()