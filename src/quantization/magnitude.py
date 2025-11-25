import torch

def compute_magnitude(model):
    sensitivity_map = {}
    print("[COMPUTING MAGNITUDE: L1 SCORES OF WEIGHTS]")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                sensitivity_map[name] = torch.sum(torch.abs(param)).item()

    return sensitivity_map

