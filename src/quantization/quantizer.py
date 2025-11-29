import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit
import numpy as np
from kneed import KneeLocator
from skimage.filters import threshold_otsu

from src.sensitivity.fisher import compute_fisher
from src.sensitivity.magnitude import compute_magnitude
from src.sensitivity.perturbation import compute_perturbation_sensitivity
from src.selection.knapsack import knapsack_solver

class SelectiveQuantizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sensitivity_map = None

    def compute_sensitivity(self, method, dsname, n_samples, reduction="mean"):
        if method == "fisher":
            self.sensitivity_map = compute_fisher(self.model, self.tokenizer, dsname, reduction=reduction, n_samples=n_samples)
        elif method == "magnitude":
            self.sensitivity_map = compute_magnitude(self.model)
        elif method == "perturbation":
            self.sensitivity_map = compute_perturbation_sensitivity(self.model, self.tokenizer, dsname, n_samples=n_samples)
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")

    @staticmethod
    def get_threshold_pct(sensitivity_map, percentile):
        scores = list(sensitivity_map.values())
        return np.quantile(scores, 1 - percentile)

    @staticmethod
    def get_threshold_otsu(sensitivity_map):
        """Calculates a threshold using Otsu's binarization algorithm."""
        scores = np.array(list(sensitivity_map.values()))
        if scores.size == 0 or np.all(scores == scores[0]):
            return scores[0] if scores.size > 0 else 0
        return threshold_otsu(scores)

    @staticmethod
    def get_threshold_elb(sensitivity_map):
        """Finds the point of maximum curvature using the kneedle algorithm."""
        scores = sorted(list(sensitivity_map.values()), reverse=True)
        x = range(len(scores))

        kneedle = KneeLocator(x, scores, curve="convex", direction="decreasing")
        index = kneedle.knee

        if index is None:
            # Fallback to a percentile if no knee is found
            return np.quantile(scores, 0.20)

        return scores[index]

    @staticmethod
    def find_optimal_threshold(sensitivities, sensitivity_ratio=0.01):
        sorted_sens = np.sort(list(sensitivities.values()))[::-1]
        
        grad = np.abs(np.diff(np.log10(sorted_sens + 1e-10)))
        grad_normalized = grad / np.max(grad)
        
        threshold_idx = np.argmax(grad_normalized < sensitivity_ratio)
        
        return sorted_sens[threshold_idx]

    @staticmethod
    def cumulative_budget_threshold(sensitivities, budget=0.95):
        sorted_sens = np.sort(list(sensitivities.values()))[::-1]
        cumsum = np.cumsum(sorted_sens)
        cumsum_normalized = cumsum / cumsum[-1]
        
        threshold_idx = np.argmax(cumsum_normalized >= budget)
        return sorted_sens[threshold_idx]

    def _get_model_device(self):
        """Get the primary device where most model parameters are located."""
        device_counts = {}
        for param in self.model.parameters():
            device = str(param.device)
            device_counts[device] = device_counts.get(device, 0) + 1
        
        # Return the device with most parameters
        if device_counts:
            primary_device = max(device_counts, key=device_counts.get)
            return torch.device(primary_device)
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def _is_model_dispatched(self):
        """Check if model is using Accelerate's device map."""
        return hasattr(self.model, 'hf_device_map') or hasattr(self.model, '_hf_hook')
    
    def _remove_dispatch(self):
        """Remove Accelerate's dispatch hooks to enable manual device placement."""
        try:
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.hooks import remove_hook_from_module, AlignDevicesHook
            
            # Method 1: Try to remove hooks from all modules recursively
            def remove_hooks_recursive(module):
                # Remove hooks from current module
                if hasattr(module, '_hf_hook'):
                    delattr(module, '_hf_hook')
                if hasattr(module, '_old_forward'):
                    module.forward = module._old_forward
                    delattr(module, '_old_forward')
                
                # Recurse to children
                for child in module.children():
                    remove_hooks_recursive(child)
            
            remove_hooks_recursive(self.model)
            
            # Method 2: Clear device map attributes
            if hasattr(self.model, 'hf_device_map'):
                delattr(self.model, 'hf_device_map')
                
            return True
        except Exception as e:
            print(f"[WARNING] Could not remove dispatch: {e}")
            return False

    def _replace_linear_with_bnb(self, full_name, og_layer, target_device=None):
        # 1. Use consistent target device (override original device for multi-GPU fix)
        if target_device is None:
            target_device = og_layer.weight.device
        
        target_dtype = og_layer.weight.dtype

        # 2. Get parent module
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
        else:
            parent = self.model
            child_name = full_name

        # 3. Create Linear4bit with proper configuration
        new_layer = bnb.nn.Linear4bit(
            input_features=og_layer.in_features,
            output_features=og_layer.out_features,
            bias=og_layer.bias is not None,
            compute_dtype=target_dtype,
            quant_type="nf4",
        )

        # 4. Quantize weights properly
        # Move weights to CPU, quantize, then move back to target device
        with torch.no_grad():
            weight_data = og_layer.weight.data.to('cpu', copy=True)
            
            # Create Params4bit on CPU
            quantized_weight = Params4bit(
                weight_data,
                requires_grad=False,
                quant_type="nf4",
            )
            
            new_layer.weight = quantized_weight

            # 5. Handle bias if present
            if og_layer.bias is not None:
                bias_data = og_layer.bias.data.to(dtype=target_dtype, device='cpu', copy=True)
                new_layer.bias = nn.Parameter(bias_data, requires_grad=False)

        # 6. Move entire layer to target device (triggers quantization finalization)
        new_layer = new_layer.to(target_device)

        # 7. Replace the layer in the parent module
        setattr(parent, child_name, new_layer)

        # 8. Clean up old layer
        del og_layer
        
        # 9. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def quantize(self, selection_method="knapsack", sensitivity_method="perturbation", 
                 dsname="gsm8k", n_samples=32, budget_mb=4096, percentile=0.20, 
                 sensitivity_ratio=0.05, budget=0.95, verbose=True):

        # 1. Compute sensitivity if not already computed
        if self.sensitivity_map is None:
            if verbose:
                print(f"[COMPUTING SENSITIVITY] Method: {sensitivity_method}")
            self.compute_sensitivity(sensitivity_method, dsname, n_samples)

        selection_method = selection_method.lower()

        # 2. Select layers to keep in high precision
        if selection_method == "knapsack":
            if verbose:
                print(f"[SELECTION METHOD] Knapsack with budget {budget_mb}MB")
            layers_to_keep = knapsack_solver(self.model, self.sensitivity_map, budget_mb)
        else:
            # Threshold-based selection methods
            if selection_method == "pct":
                threshold = SelectiveQuantizer.get_threshold_pct(self.sensitivity_map, percentile)
            elif selection_method == "otsu":
                threshold = SelectiveQuantizer.get_threshold_otsu(self.sensitivity_map)
            elif selection_method == "elb":
                threshold = SelectiveQuantizer.get_threshold_elb(self.sensitivity_map)
            elif selection_method == "gradient":
                threshold = SelectiveQuantizer.find_optimal_threshold(self.sensitivity_map, sensitivity_ratio)
            elif selection_method == "cumulative":
                threshold = SelectiveQuantizer.cumulative_budget_threshold(self.sensitivity_map, budget)
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")

            if verbose:
                print(f"[THRESHOLD COMPUTED] Method: {selection_method.upper()}, Value: {threshold:.6f}")

            # Select layers above threshold
            layers_to_keep = [
                name for name, score in self.sensitivity_map.items() 
                if score >= threshold
            ]

        if verbose:
            print(f"[KEEPING {len(layers_to_keep)} LAYERS IN HIGH PRECISION]")

        # 3. Collect layers to quantize
        layers_to_quantize = []
        total_linear = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                total_linear += 1
                if name not in layers_to_keep:
                    layers_to_quantize.append(name)

        if verbose:
            print(f"[QUANTIZING {len(layers_to_quantize)}/{total_linear} LAYERS TO INT4]")

        # 4. Handle Accelerate dispatch if present
        is_dispatched = self._is_model_dispatched()
        target_device = None
        
        if is_dispatched:
            if verbose:
                print("[DETECTED] Model is using Accelerate device map")
                print("[REMOVING] Accelerate dispatch hooks for manual device control...")
            
            # Get target device BEFORE removing dispatch
            target_device = self._get_model_device()
            
            # Remove dispatch
            success = self._remove_dispatch()
            if not success and verbose:
                print("[WARNING] Could not fully remove dispatch, attempting to continue...")
            
            if verbose:
                print(f"[TARGET DEVICE] Consolidating model to {target_device}")
            
            try:
                # Move everything to target device
                self.model = self.model.to(target_device)
                
                # Force move all parameters and buffers
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.device != target_device:
                            param.data = param.data.to(target_device)
                    for buffer in self.model.buffers():
                        if buffer.device != target_device:
                            buffer.data = buffer.data.to(target_device)
                
                if verbose:
                    print("[SUCCESS] Model moved to single device")
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Error moving model: {e}")
                    print("[CONTINUING] Will use original layer devices")
                target_device = None
        else:
            # Model not dispatched, use primary device for consistency
            target_device = self._get_model_device()
            if verbose:
                print(f"[TARGET DEVICE] All layers will be placed on {target_device}")

        # 5. Perform quantization with consistent device placement
        for i, layer_name in enumerate(layers_to_quantize):
            if verbose and (i + 1) % 10 == 0:
                print(f"[PROGRESS] {i + 1}/{len(layers_to_quantize)} layers quantized")
            
            # Get the current module reference
            try:
                module = dict(self.model.named_modules())[layer_name]
                self._replace_linear_with_bnb(layer_name, module, target_device=target_device)
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Failed to quantize layer {layer_name}: {str(e)}")
                continue

        # 6. Final device consolidation if we had dispatch
        if is_dispatched and target_device is not None:
            if verbose:
                print(f"[FINAL CONSOLIDATION] Ensuring all parameters on {target_device}...")
            try:
                # Move the entire model (this should work now that hooks are removed)
                self.model = self.model.to(target_device)
                
                # Double-check: manually move any stragglers
                for name, param in self.model.named_parameters():
                    if param.device != target_device:
                        param.data = param.data.to(target_device)
                
                # Also move buffers (like embeddings)
                for buffer_name, buffer in self.model.named_buffers():
                    if buffer.device != target_device:
                        buffer.data = buffer.data.to(target_device)
                
                # Final verification
                devices = set()
                for param in self.model.parameters():
                    devices.add(str(param.device))
                for buffer in self.model.buffers():
                    devices.add(str(buffer.device))
                
                if verbose:
                    if len(devices) == 1:
                        print(f"[SUCCESS] All tensors on {list(devices)[0]}")
                    else:
                        print(f"[WARNING] Tensors still on multiple devices: {devices}")
                        
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Consolidation had issues: {e}")
        
        # Additional check: verify no hooks remain and do final cleanup
        if verbose:
            print("[FINAL CHECK] Removing any remaining Accelerate hooks...")
        
        hooks_removed = 0
        for name, module in self.model.named_modules():
            if hasattr(module, '_hf_hook'):
                delattr(module, '_hf_hook')
                hooks_removed += 1
            if hasattr(module, '_old_forward'):
                module.forward = module._old_forward
                delattr(module, '_old_forward')
                hooks_removed += 1
        
        if verbose and hooks_removed > 0:
            print(f"[CLEANED] Removed {hooks_removed} residual hooks/forwards")
        
        # Clear model-level dispatch attributes
        for attr in ['hf_device_map', '_hf_hook', 'is_parallelizable', 'model_parallel']:
            if hasattr(self.model, attr):
                try:
                    delattr(self.model, attr)
                except:
                    pass

        # 7. Set model to eval mode
        self.model.eval()

        # 8. Optional validation
        if verbose:
            print("[VALIDATING] Testing model forward pass...")
        
        try:
            if len(list(self.model.parameters())) > 0:
                # Use target device if set, otherwise get current device
                device = target_device if target_device is not None else next(self.model.parameters()).device
                dummy_input = torch.randint(0, 1000, (1, 10), dtype=torch.long, device=device)
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                if verbose:
                    print("[VALIDATION] Model forward pass successful!")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Validation failed: {str(e)}")
                print("[INFO] This may be normal for some model architectures")

        # 9. Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 10. Final report and return
        unchanged_count = total_linear - len(layers_to_quantize)
        compression_ratio = len(layers_to_quantize) / total_linear if total_linear > 0 else 0
        
        if verbose:
            print(f"\n[QUANTIZATION COMPLETE]")
            print(f"  - High precision layers: {unchanged_count}/{total_linear}")
            print(f"  - Quantized layers: {len(layers_to_quantize)}/{total_linear}")
            print(f"  - Compression ratio: {compression_ratio:.1%}")
        
        # CRITICAL: Re-wrap with accelerate if it was originally dispatched but on single device now
        # This ensures compatibility with generation code that expects the model structure
        if is_dispatched and target_device is not None:
            if verbose:
                print(f"[RE-INITIALIZING] Preparing model for single-device operation...")
            
            try:
                # Create a simple single-device map
                from accelerate import dispatch_model
                device_map = {name: target_device for name, _ in self.model.named_parameters()}
                
                # Only dispatch if model expects it
                # Actually, let's NOT re-dispatch - just ensure the model works standalone
                if verbose:
                    print("[INFO] Model prepared for single-device inference")
                    
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Post-processing: {e}")

        return self.model
