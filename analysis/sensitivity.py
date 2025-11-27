import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import Config

def parse_meta(filename):
    parts = filename[:-5].split('_')
    if parts[0] == 'fisher':
        return {'kind':'fisher','dataset':parts[1],'samples': int(parts[2]) if len(parts)>2 and parts[2].isdigit() else None}
    if parts[0] == 'magnitude':
        return {'kind':'magnitude','dataset':parts[1]}
    return {}

def organize_and_preprocess_data():
    maps_dir = Config.MAPS_DIR

    json_files = [f for f in os.listdir(maps_dir) if f.endswith('.json') and f != 'summary.json']

    data_dict = {}

    for file in json_files:
        file_path = os.path.join(maps_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        data_dict[file] = data

    return data_dict

def aggregate_fisher_per_layer(fisher_data):
    agg = defaultdict(lambda: defaultdict(list))

    for fname, data in fisher_data.items():
        if fname.startswith('fisher_'):
            meta = parse_meta(fname)
            if meta.get('kind') == 'fisher':
                ds = meta['dataset']
                for k, v in data.items():
                    agg[ds][k].append(v)

    fisher_by_dataset = {}
    for ds, pdict in agg.items():
        fisher_by_dataset[ds] = {k: sum(vals)/len(vals) for k, vals in pdict.items()}

    layer_aggregated = defaultdict(lambda: {"fisher_sum": 0, "fisher_mean": 0, "count": 0})

    for dataset, data in fisher_by_dataset.items():
        for param_name, fisher_value in data.items():
            if '.weight' in param_name:
                layer_name = param_name.rsplit('.', 1)[0]
            else:
                layer_name = param_name

            layer_aggregated[layer_name]["fisher_sum"] += fisher_value
            layer_aggregated[layer_name]["count"] += 1

    for layer_name in layer_aggregated:
        count = layer_aggregated[layer_name]["count"]
        if count > 0:
            layer_aggregated[layer_name]["fisher_mean"] = layer_aggregated[layer_name]["fisher_sum"] / count

    return dict(layer_aggregated), fisher_by_dataset

def identify_top_k_sensitive_layers(layer_data, k=10):
    sorted_layers = sorted(layer_data.items(), key=lambda x: x[1]["fisher_mean"], reverse=True)
    return sorted_layers[:k]

def group_by_module_type(layer_data):
    module_groups = defaultdict(list)

    for layer_name, data in layer_data.items():
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
    module_stats = {}

    for module_type, layers in module_groups.items():
        fisher_means = [layer[1]["fisher_mean"] for layer in layers]
        
        module_stats[module_type] = {
            "mean_fisher_mean": np.mean(fisher_means) if fisher_means else 0,
            "max_fisher_mean": np.max(fisher_means) if fisher_means else 0,
            "min_fisher_mean": np.min(fisher_means) if fisher_means else 0,
            "std_fisher_mean": np.std(fisher_means) if fisher_means else 0,
            "count": len(layers)
        }

    return module_stats

def magnitude_vs_sensitivity_analysis(fisher_by_dataset, magnitude_data):
    result = []

    for dataset, fisher_values in fisher_by_dataset.items():
        magnitude_file = f"magnitude_{dataset}.json"

        if magnitude_file in magnitude_data:
            mag_values = magnitude_data[magnitude_file]

            common_params = set(fisher_values.keys()) & set(mag_values.keys())

            for param in common_params:
                result.append({
                    'param_name': param,
                    'fisher_value': fisher_values[param],
                    'magnitude_value': mag_values[param],
                    'dataset': dataset
                })

    return result

def create_parameter_ranking_per_dataset(fisher_by_dataset):
    all_params_by_dataset = {}

    for dataset, data in fisher_by_dataset.items():
        dataset_params = []
        for param_name, fisher_value in data.items():
            dataset_params.append({
                'param_name': param_name,
                'fisher_value': fisher_value,
                'dataset': dataset
            })
        dataset_params.sort(key=lambda x: x['fisher_value'], reverse=True)
        all_params_by_dataset[dataset] = dataset_params

    return all_params_by_dataset

def generate_zones_per_dataset(param_scores, top_percent):
    names = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(names)
    a = int(n * top_percent)
    for i,(name,score) in enumerate(names):
        zone = 'A' if i < a else 'B' if i < 2*a else 'C'
        yield name, score, zone

def generate_quantization_zones_per_dataset(fisher_by_dataset, top_percent=0.1):
    all_zones = {}

    for dataset, data in fisher_by_dataset.items():
        zones = []
        for param_name, score, zone in generate_zones_per_dataset(data, top_percent):
            zones.append({
                'param_name': param_name,
                'fisher_value': score,
                'zone': zone,
                'dataset': dataset
            })
        all_zones[dataset] = zones

    return all_zones

def generate_reports(aggregated_data, module_stats, param_rankings_by_dataset, quantization_zones, top_k, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    top_layers = identify_top_k_sensitive_layers(aggregated_data, k=top_k)
    top_layers_df = pd.DataFrame([
        {
            'layer_name': layer[0],
            'fisher_mean': layer[1]['fisher_mean'],
            'fisher_sum': layer[1]['fisher_sum'],
            'parameter_count': layer[1]['count']
        }
        for layer in top_layers
    ])

    module_stats_df = pd.DataFrame.from_dict(module_stats, orient='index')
    module_stats_df.index.name = 'module_type'

    all_param_rankings = []
    for dataset, rankings in param_rankings_by_dataset.items():
        all_param_rankings.extend(rankings)

    param_rankings_df = pd.DataFrame(all_param_rankings)

    all_zones = []
    for dataset, zones in quantization_zones.items():
        all_zones.extend(zones)

    zones_df = pd.DataFrame(all_zones)

    top_layers_df.to_csv(os.path.join(output_dir, 'top_sensitive_layers.csv'), index=False)
    module_stats_df.to_csv(os.path.join(output_dir, 'module_statistics.csv'))
    param_rankings_df.to_csv(os.path.join(output_dir, 'parameter_rankings.csv'), index=False)
    zones_df.to_csv(os.path.join(output_dir, 'quantization_zones.csv'), index=False)

    for dataset, rankings in param_rankings_by_dataset.items():
        df = pd.DataFrame(rankings)
        df.to_csv(os.path.join(output_dir, f'parameter_rankings_{dataset}.csv'), index=False)

    for dataset, zones in quantization_zones.items():
        df = pd.DataFrame(zones)
        df.to_csv(os.path.join(output_dir, f'quantization_zones_{dataset}.csv'), index=False)

    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Top {top_k} Most Sensitive Layers:\n")
        for i, layer in enumerate(top_layers):
            f.write(f"{i+1:2d}. {layer[0]}: Fisher Mean = {layer[1]['fisher_mean']:.6f}\n")

        f.write(f"\nModule Type Statistics:\n")
        for module_type, stats in module_stats.items():
            f.write(f"{module_type}: Mean Fisher = {stats['mean_fisher_mean']:.6f}, "
                    f"Max Fisher = {stats['max_fisher_mean']:.6f}, Count = {stats['count']}\n")

    return top_layers_df, module_stats_df, param_rankings_df, zones_df

def run_full_analysis(model_name, top_k=10, top_percent=0.1):
    Config.set_model(model_name)
    
    print(f"[MODEL] {model_name}")
    print("[LOADING] All Fisher and magnitude JSON results")
    all_data = organize_and_preprocess_data()

    fisher_data = {k: v for k, v in all_data.items() if k.startswith('fisher_')}
    magnitude_data = {k: v for k, v in all_data.items() if k.startswith('magnitude_')}

    print("[AGGREGATING] Fisher values per layer and per dataset")
    layer_aggregated, fisher_by_dataset = aggregate_fisher_per_layer(fisher_data)

    print("[GROUPING] Parameters by module type")
    module_groups = group_by_module_type(layer_aggregated)
    module_stats = compute_module_statistics(module_groups)

    print("[ANALYZING] Magnitude vs Sensitivity")
    mag_vs_sens = magnitude_vs_sensitivity_analysis(fisher_by_dataset, magnitude_data)

    print("[RANKING] Parameters by sensitivity (per dataset)")
    param_rankings_by_dataset = create_parameter_ranking_per_dataset(fisher_by_dataset)

    print("[GENERATING] Quantization zones (per dataset)")
    quantization_zones = generate_quantization_zones_per_dataset(fisher_by_dataset, top_percent=top_percent)

    print("[GENERATING] Reports")
    output_dir = os.path.join(Config.RESULTS_DIR, 'reports', model_name)
    top_layers_df, module_stats_df, param_rankings_df, zones_df = generate_reports(
        layer_aggregated, module_stats, param_rankings_by_dataset, quantization_zones, top_k, output_dir
    )

    print(f"[SAVED] Analysis reports to {output_dir}")
    print("[COMPLETE] Sensitivity analysis finished")

    return {
        'top_layers': top_layers_df,
        'module_stats': module_stats_df,
        'param_rankings': param_rankings_df,
        'quantization_zones': zones_df,
        'layer_aggregated': layer_aggregated,
        'module_groups': module_groups,
        'magnitude_vs_sensitivity': mag_vs_sens,
        'fisher_by_dataset': fisher_by_dataset
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["qwen", "qwen_3b", "phi", "stablelm", "llama_3b"], help="Name of the model to analyze")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top sensitive layers to report")
    parser.add_argument("--top_percent", type=float, default=0.1, help="Top percentage of layers to include in quantization zone 'A'")
    args = parser.parse_args()
    
    run_full_analysis(model_name=args.model_name, top_k=args.top_k, top_percent=args.top_percent)
