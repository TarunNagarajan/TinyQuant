import torch
import numpy as np
from scipy.stats import trim_mean
from tqdm import tqdm
from src.config import Config
from src.data.loader import get_calibration_data


def compute_fisher(model, tokenizer, dsname, reduction="mean", n_samples=None, 
                   clip_percentile=99.0, clip_samples=32):
    """
    Compute Fisher Information Matrix diagonal approximation for each layer.
    
    Uses adaptive gradient clipping to ensure robustness against outlier samples.
    
    Args:
        model: The language model (in eval mode)
        tokenizer: Tokenizer for the model
        dsname: Dataset name for calibration ('gsm8k', 'wikitext', 'math')
        reduction: 'mean' or 'sum' - how to aggregate Fisher scores across samples
        n_samples: Number of calibration samples (default: Config.CALIBRATION_SAMPLES)
        clip_percentile: Percentile for adaptive gradient clipping (default: 99.0)
                        Set to None to disable clipping entirely
        clip_samples: Number of samples to use for estimating clip threshold (default: 32)
    
    Returns:
        sensitivity_map: Dict mapping module names to Fisher information scores
                        (normalized by parameter count)
    """
    assert reduction in ("mean", "sum"), f"reduction must be 'mean' or 'sum', got {reduction}"
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get actual device model is on (handle multi-GPU setups)
    model_device = next(model.parameters()).device
    
    print(f"[{dsname}] [COMPUTING FISHER INFORMATION]")
    print(f"[{dsname}] [MODEL DEVICE] {model_device}")
    print(f"[{dsname}] [REDUCTION] {reduction}")
    
    # Load calibration data
    raw_samples = get_calibration_data(dsname, n_samples=n_samples)
    print(f"[{dsname}] [SAMPLES] {len(raw_samples)} calibration samples")
    
    # Initialize storage
    sensitivity_map = {}
    
    # ==================================================================
    # PHASE 1: ADAPTIVE CLIPPING THRESHOLD ESTIMATION
    # ==================================================================
    max_grad_norm = None
    
    if clip_percentile is not None:
        print(f"\n[{dsname}] [PHASE 1/2] Estimating adaptive clipping threshold...")
        print(f"[{dsname}] [USING] {min(clip_samples, len(raw_samples))} samples for threshold estimation")
        
        all_grad_norms = []
        samples_processed_phase1 = 0
        
        for i, text in enumerate(tqdm(raw_samples[:clip_samples], 
                                       desc="Collecting gradient statistics", 
                                       leave=False)):
            try:
                # Tokenize
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=2048
                )
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                # Forward + backward
                model.zero_grad(set_to_none=True)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Skip non-finite losses
                if not torch.isfinite(loss):
                    del inputs, outputs, loss
                    continue
                
                loss.backward()
                
                # Collect gradient norms per parameter
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and "weight" in name:
                            grad_norm = param.grad.norm().item()
                            if np.isfinite(grad_norm) and grad_norm > 0:
                                all_grad_norms.append(grad_norm)
                
                samples_processed_phase1 += 1
                
                # Cleanup
                model.zero_grad(set_to_none=True)
                del inputs, outputs, loss
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARNING] OOM in phase 1 sample {i}, skipping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            
            except Exception as e:
                print(f"[WARNING] Error in phase 1 sample {i}: {str(e)}")
                continue
        
        # Compute adaptive threshold
        if all_grad_norms:
            grad_norms_array = np.array(all_grad_norms)
            max_grad_norm = np.percentile(grad_norms_array, clip_percentile)
            
            print(f"[{dsname}] [PHASE 1 COMPLETE]")
            print(f"[{dsname}] [PROCESSED] {samples_processed_phase1} samples")
            print(f"[{dsname}] [GRADIENT STATISTICS]")
            print(f"           Min:    {grad_norms_array.min():.2e}")
            print(f"           Max:    {grad_norms_array.max():.2e}")
            print(f"           Mean:   {grad_norms_array.mean():.2e}")
            print(f"           Median: {np.median(grad_norms_array):.2e}")
            print(f"           95th:   {np.percentile(grad_norms_array, 95):.2e}")
            print(f"           99th:   {np.percentile(grad_norms_array, 99):.2e}")
            print(f"[{dsname}] [CLIP THRESHOLD] {max_grad_norm:.4f} (at {clip_percentile}th percentile)")
            
            # Sanity check
            if max_grad_norm < 1e-6:
                print(f"[WARNING] Clip threshold is very small ({max_grad_norm:.2e}). This may indicate an issue.")
            if max_grad_norm > 1e6:
                print(f"[WARNING] Clip threshold is very large ({max_grad_norm:.2e}). Consider investigating.")
        else:
            print(f"[WARNING] Could not collect gradient norms in phase 1. Disabling clipping.")
            max_grad_norm = None
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[{dsname}] [CLIPPING DISABLED] clip_percentile=None")
    
    # ==================================================================
    # PHASE 2: FISHER INFORMATION COMPUTATION
    # ==================================================================
    print(f"\n[{dsname}] [PHASE 2/2] Computing Fisher Information...")
    
    processed = 0
    failed = 0
    clipped_count = 0
    gradient_stats = []  # For final diagnostics
    
    for i, text in enumerate(tqdm(raw_samples, desc="Computing Fisher", leave=False)):
        try:
            # Tokenize and move to device
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Zero gradients
            model.zero_grad(set_to_none=True)
            
            # Forward pass with labels for loss computation
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss on sample {i}, skipping")
                failed += 1
                del inputs, outputs, loss
                continue
            
            # Backward pass
            loss.backward()
            
            # Compute Fisher information (squared gradients)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # Skip non-weight parameters
                    if param.grad is None or "weight" not in name:
                        continue
                    
                    # Get gradient
                    grad = param.grad.detach()
                    
                    # Check for invalid gradients
                    if not torch.isfinite(grad).all():
                        print(f"[WARNING] Non-finite gradient in {name} on sample {i}, skipping layer")
                        continue
                    
                    # Compute gradient norm
                    grad_norm = grad.norm().item()
                    gradient_stats.append(grad_norm)
                    
                    # Apply adaptive clipping if enabled
                    if max_grad_norm is not None and grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / (grad_norm + 1e-8))
                        clipped_count += 1
                    
                    # Compute squared gradient sum
                    grad_sq = grad.float().square().sum().item()
                    
                    # CRITICAL: Normalize by parameter count
                    # This gives us Fisher Information "density" - importance per parameter
                    # Without this, large layers dominate simply due to having more parameters
                    fisher_density = grad_sq / param.numel()
                    
                    # Accumulate
                    module_name = name.replace(".weight", "")
                    if module_name not in sensitivity_map:
                        sensitivity_map[module_name] = 0.0
                    sensitivity_map[module_name] += fisher_density
            
            processed += 1
            
            # Cleanup
            model.zero_grad(set_to_none=True)
            del inputs, outputs, loss
            
            # Periodic memory cleanup
            if torch.cuda.is_available() and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            # Catch OOM errors
            if "out of memory" in str(e).lower():
                print(f"[ERROR] OOM on sample {i}, skipping. Consider reducing max_length or batch size.")
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
    
    # ==================================================================
    # FINAL PROCESSING AND VALIDATION
    # ==================================================================
    
    # Check if we processed enough samples
    if processed == 0:
        raise RuntimeError(
            f"No samples were successfully processed! "
            f"Failed: {failed}/{len(raw_samples)}. "
            f"Check your model, tokenizer, and dataset."
        )
    
    if processed < len(raw_samples) * 0.5:
        print(f"[WARNING] Only processed {processed}/{len(raw_samples)} samples ({100*processed/len(raw_samples):.1f}%). "
              f"Results may be unreliable.")
    
    # Apply reduction
    if reduction == "mean":
        for name in sensitivity_map:
            sensitivity_map[name] /= processed
    
    # ==================================================================
    # DIAGNOSTICS AND REPORTING
    # ==================================================================
    
    print(f"\n[{dsname}] [FISHER COMPUTATION COMPLETE]")
    print(f"[{dsname}] [SAMPLES] {processed}/{len(raw_samples)} processed successfully")
    if failed > 0:
        print(f"[{dsname}] [FAILED] {failed} samples")
    if max_grad_norm is not None and clipped_count > 0:
        total_grads = processed * len(sensitivity_map)
        clip_rate = 100 * clipped_count / total_grads
        print(f"[{dsname}] [CLIPPED] {clipped_count}/{total_grads} gradients ({clip_rate:.2f}%) exceeded threshold {max_grad_norm:.4f}")
    
    # Gradient statistics
    if gradient_stats:
        grad_array = np.array(gradient_stats)
        print(f"\n[{dsname}] [GRADIENT STATISTICS] (across all samples & layers)")
        print(f"           Min:    {grad_array.min():.2e}")
        print(f"           Max:    {grad_array.max():.2e}")
        print(f"           Mean:   {grad_array.mean():.2e}")
        print(f"           Median: {np.median(grad_array):.2e}")
        print(f"           Std:    {grad_array.std():.2e}")
    
    # Fisher score statistics
    if sensitivity_map:
        scores = np.array(list(sensitivity_map.values()))
        print(f"\n[{dsname}] [FISHER SCORES] (final sensitivity values)")
        print(f"           Layers:  {len(scores)}")
        print(f"           Min:     {scores.min():.2e}")
        print(f"           Max:     {scores.max():.2e}")
        print(f"           Mean:    {scores.mean():.2e}")
        print(f"           Median:  {np.median(scores):.2e}")
        print(f"           Std:     {scores.std():.2e}")
        
        # Check for potential issues
        cv = scores.std() / (scores.mean() + 1e-10)  # Coefficient of variation
        if cv > 10:
            print(f"[WARNING] High coefficient of variation ({cv:.1f}). Scores have high variance.")
        
        zero_scores = np.sum(scores == 0)
        if zero_scores > 0:
            print(f"[WARNING] {zero_scores}/{len(scores)} layers have zero Fisher information")
        
        # Check for extreme outliers
        q99 = np.percentile(scores, 99)
        extreme_outliers = np.sum(scores > 10 * q99)
        if extreme_outliers > 0:
            print(f"[WARNING] {extreme_outliers} layers have scores >10x the 99th percentile. "
                  f"Consider stricter clipping.")
    
    # Validate results
    _validate_fisher_scores(sensitivity_map, model)
    
    return sensitivity_map


def _validate_fisher_scores(sensitivity_map, model):
    """
    Internal validation function to check Fisher scores are reasonable.
    
    Performs sanity checks:
    1. All scores are non-negative (required by definition)
    2. Not all scores are zero (would indicate computation failure)
    3. Expected number of layers present
    4. Score distribution is reasonable
    5. Module names don't contain '.weight' suffix
    """
    if not sensitivity_map:
        raise ValueError("Fisher sensitivity map is empty!")
    
    scores = np.array(list(sensitivity_map.values()))
    
    # Check 1: Non-negative (Fisher information must be ≥ 0)
    if np.any(scores < 0):
        negative_layers = [name for name, score in sensitivity_map.items() if score < 0]
        raise ValueError(f"Fisher scores must be non-negative! Found negative scores in: {negative_layers[:5]}")
    
    # Check 2: Not all zeros
    if np.all(scores == 0):
        raise ValueError("All Fisher scores are zero - computation failed!")
    
    # Check 3: Expected number of layers
    expected_linear_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    if len(sensitivity_map) < expected_linear_layers * 0.7:  # Allow 30% margin for flexibility
        print(f"[WARNING] Expected ~{expected_linear_layers} layers, got {len(sensitivity_map)}. "
              f"Some layers may be missing.")
    
    # Check 4: Reasonable distribution (not too extreme)
    if scores.max() > 0:
        dynamic_range = scores.max() / (scores.min() + 1e-10)
        if dynamic_range > 1e10:
            print(f"[WARNING] Extreme dynamic range in scores ({dynamic_range:.2e}). "
                  f"This may cause numerical issues.")
    
    # Check 5: Layer name format
    for name in sensitivity_map.keys():
        if ".weight" in name:
            raise ValueError(
                f"Module name should not contain '.weight' suffix: {name}\n"
                f"Fisher computation should strip '.weight' from parameter names."
            )
    
    print(f"[{model.__class__.__name__}] [VALIDATION] Fisher scores passed sanity checks")


def compare_fisher_methods(model, tokenizer, dsname, n_samples=128):
    """
    Utility function to compare Fisher computation with different settings.
    Useful for debugging and finding optimal hyperparameters.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer
        dsname: Dataset name
        n_samples: Number of samples for comparison
    
    Returns:
        dict: Results from different configurations
    """
    results = {}
    
    print("="*80)
    print("COMPARING FISHER COMPUTATION METHODS")
    print("="*80)
    
    # Test 1: No clipping
    print("\n[TEST 1] No gradient clipping")
    results['no_clip'] = compute_fisher(
        model, tokenizer, dsname, 
        n_samples=n_samples, 
        clip_percentile=None
    )
    
    # Test 2: Conservative clipping (95th percentile)
    print("\n[TEST 2] Conservative clipping (95th percentile)")
    results['clip_95'] = compute_fisher(
        model, tokenizer, dsname, 
        n_samples=n_samples, 
        clip_percentile=95.0
    )
    
    # Test 3: Standard clipping (99th percentile)
    print("\n[TEST 3] Standard clipping (99th percentile)")
    results['clip_99'] = compute_fisher(
        model, tokenizer, dsname, 
        n_samples=n_samples, 
        clip_percentile=99.0
    )
    
    # Test 4: Lenient clipping (99.5th percentile)
    print("\n[TEST 4] Lenient clipping (99.5th percentile)")
    results['clip_99.5'] = compute_fisher(
        model, tokenizer, dsname, 
        n_samples=n_samples, 
        clip_percentile=99.5
    )
    
    # Compare correlations
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    from scipy.stats import spearmanr
    
    methods = list(results.keys())
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            # Get common layers
            common_layers = set(results[method1].keys()) & set(results[method2].keys())
            
            scores1 = [results[method1][l] for l in common_layers]
            scores2 = [results[method2][l] for l in common_layers]
            
            correlation, p_value = spearmanr(scores1, scores2)
            print(f"{method1:15s} vs {method2:15s}: ρ = {correlation:.4f} (p={p_value:.2e})")
    
    return results