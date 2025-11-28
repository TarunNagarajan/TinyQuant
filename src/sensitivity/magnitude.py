import torch

def compute_magnitude(model):
    # Squared L2 norm (standard in quantization literature)
    sensitivity_map = {}
    print("[COMPUTING MAGNITUDE: SQUARED L2 NORM]")
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name:
                module_name = name.replace(".weight", "")
                sensitivity_map[module_name] = param.pow(2).sum().item()
    
    return sensitivity_map
