import torch
import torch.nn as nn

def get_param_size(param):
    """
    Calculates the memory size of a parameter in megabytes.
    """
    return param.numel() * param.element_size() / (1024 * 1024)

def get_module_memory_cost(module, precision):
    """
    Calculates the memory cost of a module at a given precision.
    """
    cost = 0
    if precision == "fp16":
        for param in module.parameters():
            cost += get_param_size(param)
    elif precision == "int4":
        # For INT4, we assume a 4-bit representation.
        # This is a simplification; actual memory can be higher due to scaling factors.
        for param in module.parameters():
            cost += param.numel() / 2 / (1024 * 1024) # 4 bits = 0.5 bytes
    return cost

def knapsack_solver(model, sensitivity_map, budget_mb):
    """
    Selects layers to keep in high precision using a greedy knapsack-like approach.

    Args:
        model: The model to analyze.
        sensitivity_map: A dictionary mapping layer names to their sensitivity scores.
        budget_mb: The memory budget in megabytes.

    Returns:
        A list of layer names to keep in high precision.
    """
    layer_data = []
    for name, module in model.named_modules():
        if name in sensitivity_map and isinstance(module, nn.Linear):
            sensitivity = sensitivity_map[name]
            
            # Cost is the difference between FP16 and INT4
            cost_fp16 = get_module_memory_cost(module, "fp16")
            cost_int4 = get_module_memory_cost(module, "int4")
            cost_to_keep = cost_fp16 - cost_int4
            
            if cost_to_keep > 0:
                density = sensitivity / cost_to_keep
                layer_data.append({
                    "name": name,
                    "sensitivity": sensitivity,
                    "cost": cost_to_keep,
                    "density": density
                })

    # Sort layers by density in descending order
    layer_data.sort(key=lambda x: x["density"], reverse=True)

    keep_list = []
    current_cost = 0
    for layer in layer_data:
        if current_cost + layer["cost"] <= budget_mb:
            keep_list.append(layer["name"])
            current_cost += layer["cost"]

    return keep_list
