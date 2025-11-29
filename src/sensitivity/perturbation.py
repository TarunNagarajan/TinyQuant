import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.config import Config
from src.data.loader import get_calibration_data

def fake_quantize_int4(weight):
    """
    Simulates INT4 quantization noise.
    Uses simple AbsMax quantization (Symmetric).
    """
    # 1. Calculate Scale (Per-Tensor for speed, or use dim=0 for Per-Channel)
    # Using Per-Channel usually matches 4-bit loaders (bitsandbytes) better
    scale = weight.abs().amax(dim=1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-5) # Prevent div by zero

    # 2. Quantize
    w_int4 = (weight / scale).round().clamp(-8, 7)

    # 3. Dequantize (The "Fake" part)
    w_dequant = w_int4 * scale
    
    return w_dequant

def compute_perturbation_sensitivity(model, tokenizer, dsname, n_samples=32):
    """
    Computes sensitivity by MEASURING the output MSE between FP16 and Simulated INT4.
    
    Args:
        n_samples: 32 is usually statistically sufficient for sensitivity analysis.
                   Using more is slower but slightly more stable.
    """
    model.eval()
    sensitivity_map = {}
    
    # Store hooks handles to remove them later
    hooks = []
    # Temporary storage for inputs captured during forward pass
    layer_inputs = {}

    def get_input_hook(name):
        def hook(module, input, output):
            # Capture the input tuple (input[0] is the tensor)
            # Detach to save memory, strictly minimal storage
            layer_inputs[name] = input[0].detach() 
        return hook

    # 1. Register Hooks on all Linear layers
    # We only care about Linear layers (Projections) for quantization
    print(f"[{dsname}] [SETUP] Registering hooks for Perturbation Analysis...")
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name)
            hooks.append(module.register_forward_hook(get_input_hook(name)))

    raw_samples = get_calibration_data(dsname, n_samples=n_samples)
    print(f"[{dsname}] [COMPUTING PERTURBATION SENSITIVITY] (Samples: {len(raw_samples)})")

    # 2. The Analysis Loop
    processed = 0
    for i, text in enumerate(tqdm(raw_samples, desc="Measuring MSE")):
        # A. Prepare Input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(Config.DEVICE)

        # B. Run Forward Pass (Hooks capture inputs)
        with torch.no_grad():
            model(**inputs)

        # C. Measure Damage Per Layer
        for name in target_modules:
            module = dict(model.named_modules())[name]
            
            if name not in layer_inputs:
                continue # Should not happen if hook fired
            
            inp = layer_inputs[name]
            
            with torch.no_grad():
                inp = inp.to(module.weight.device)
                # 1. Ground Truth Output (FP16/BF16)
                # We re-compute this locally to ensure we compare against the exact same input
                out_gt = module(inp)
                
                # 2. Fake Quantized Output (Simulated INT4)
                w_orig = module.weight.data
                w_quant = fake_quantize_int4(w_orig)
                
                # Manual linear pass: y = xA^T + b
                out_quant = F.linear(inp, w_quant, module.bias)
                
                # 3. Compute MSE (The Sensitivity Score)
                # We calculate standard MSE
                mse = (out_gt - out_quant).pow(2).mean().item()
                
                # OPTIONAL: Normalize by output magnitude (Relative MSE)
                # This helps if layers have vastly different scales. 
                # For basic selection, raw MSE is often fine, but Relative is safer.
                out_norm = out_gt.pow(2).mean().item() + 1e-6
                score = mse / out_norm

                sensitivity_map[name] = sensitivity_map.get(name, 0.0) + score

        # D. Cleanup to prevent VRAM explosion
        layer_inputs.clear()
        processed += 1
        
        # Periodic Cache Clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Remove Hooks
    for h in hooks:
        h.remove()

    # 4. Normalize Score by number of samples
    for name in sensitivity_map:
        sensitivity_map[name] /= processed

    if sensitivity_map: # Check if map is not empty
        scores_values = list(sensitivity_map.values())
        print(f"[DEBUG PERTURBATION] Scores - Min: {min(scores_values):.2e}, Max: {max(scores_values):.2e}, Mean: {sum(scores_values)/len(scores_values):.2e}")

    return sensitivity_map
