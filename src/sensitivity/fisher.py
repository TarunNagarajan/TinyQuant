import torch
import numpy as np
from tqdm import tqdm
from src.config import Config
from src.data.loader import get_calibration_data

def compute_fisher(model, tokenizer, dsname, reduction="mean", n_samples=None, max_grad_norm=10.0):
    """
    Compute Fisher Information Matrix diagonal approximation for each layer.
    
    This measures the expected squared gradient of the loss w.r.t. each parameter,
    which approximates how much the model's predictions depend on each parameter.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        dsname: Dataset name for calibration
        reduction: 'mean' or 'sum' - how to aggregate across samples
        n_samples: Number of calibration samples (default: Config.CALIBRATION_SAMPLES)
        max_grad_norm: Maximum gradient norm for clipping outliers (default: 10.0)
    
    Returns:
        sensitivity_map: Dict mapping module names to Fisher information scores
    """
    assert reduction in ("mean", "sum"), f"reduction must be 'mean' or 'sum', got {reduction}"
    
    model.eval()
    
    # Get actual device model is on (handle multi-GPU)
    model_device = next(model.parameters()).device
    print(f"[{dsname}] [COMPUTING FISHER INFORMATION]")
    print(f"[{dsname}] [MODEL DEVICE] {model_device}")
    
    sensitivity_map = {}
    raw_samples = get_calibration_data(dsname, n_samples=n_samples)
    
    processed = 0
    failed = 0
    gradient_norms = []  # Track gradient statistics for diagnostics
    
    for i, text in enumerate(tqdm(raw_samples, desc="Computing Fisher", leave=False)):
        try:
            # Tokenize and move to correct device
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Zero gradients
            model.zero_grad(set_to_none=True)  # set_to_none=True is more memory efficient
            
            # Forward pass with labels for loss computation
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss on sample {i}, skipping")
                failed += 1
                continue
            
            # Backward pass
            loss.backward()
            
            # Compute Fisher information (squared gradients) with gradient clipping
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    
                    if "weight" not in name:
                        continue  # Only process weight parameters
                    
                    # Get gradient and check for issues
                    grad = param.grad.detach()
                    
                    if not torch.isfinite(grad).all():
                        print(f"[WARNING] Non-finite gradient in {name} on sample {i}, skipping layer")
                        continue
                    
                    # Compute gradient norm for this parameter (for diagnostics)
                    grad_norm = grad.norm().item()
                    gradient_norms.append(grad_norm)
                    
                    # Apply gradient clipping if needed (per-parameter)
                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / (grad_norm + 1e-8))
                    
                    # Compute squared gradient sum
                    grad_sq = grad.float().square().sum().item()
                    
                    # Normalize by parameter count (Fisher Information density)
                    # This prevents large layers from dominating just due to size
                    fisher_density = grad_sq / param.numel()
                    
                    # Accumulate
                    module_name = name.replace(".weight", "")
                    sensitivity_map[module_name] = sensitivity_map.get(module_name, 0.0) + fisher_density
            
            processed += 1
            
            # Cleanup
            model.zero_grad(set_to_none=True)
            del inputs, outputs, loss
            
            # Periodic memory cleanup
            if torch.cuda.is_available() and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            # Catch OOM or other runtime errors
            if "out of memory" in str(e).lower():
                print(f"[ERROR] OOM on sample {i}, skipping. Consider reducing max_length.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"[ERROR] Runtime error on sample {i}: {str(e)}")
            failed += 1
            continue
        
        except Exception as e:
            print(f"[ERROR] Unexpected error on sample {i}: {str(e)}")
            failed += 1
            continue
    
    # Final reduction
    if processed == 0:
        raise RuntimeError("No samples were successfully processed! Check your data and model.")
    
    if reduction == "mean":
        for name in sensitivity_map:
            sensitivity_map[name] /= processed
    
    # Print diagnostics
    print(f"[{dsname}] [FISHER COMPLETE]")
    print(f"[{dsname}] [PROCESSED] {processed}/{len(raw_samples)} samples")
    if failed > 0:
        print(f"[{dsname}] [FAILED] {failed} samples")
    
    if gradient_norms:
        grad_norms_array = np.array(gradient_norms)
        print(f"[{dsname}] [GRADIENT STATS] Min: {grad_norms_array.min():.2e}, "
              f"Max: {grad_norms_array.max():.2e}, Mean: {grad_norms_array.mean():.2e}, "
              f"Median: {np.median(grad_norms_array):.2e}")
    
    if sensitivity_map:
        scores = np.array(list(sensitivity_map.values()))
        print(f"[{dsname}] [FISHER SCORES] Min: {scores.min():.2e}, "
              f"Max: {scores.max():.2e}, Mean: {scores.mean():.2e}, "
              f"Median: {np.median(scores):.2e}")
        
        # Check for potential issues
        if scores.max() / (scores.mean() + 1e-10) > 1000:
            print(f"[WARNING] Very high score variance detected. Consider using gradient clipping.")
        
        zero_scores = np.sum(scores == 0)
        if zero_scores > 0:
            print(f"[INFO] {zero_scores}/{len(scores)} layers have zero Fisher information")
    
    return sensitivity_map
