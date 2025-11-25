import torch

# calculate the L1 norm of the weights, as a standard magnitude pruning metric
def compute_magnitude(model):
    sensitivity_map = {}
    print("[COMPUTING L1 SCORES OF WEIGHTS]")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "weight":
                sensitivity_map[name] = param.abs().sum().item()

    return sensitivity_map

