import torch

def compute_magnitude(model):
    sensitivity_map = {}
    print("[COMPUTING MAGNITUDE: L1 SCORES OF WEIGHTS]")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name:
                sensitivity_map[name] = param.abs().sum().item()

    return sensitivity_map
