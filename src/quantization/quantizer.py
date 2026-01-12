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

    def compute_sensitivity(self, method, dsname, n_samples, reduction="mean", 
                       fisher_clip_percentile=99.0, fisher_clip_samples=32):
        if method == "fisher":
            self.sensitivity_map = compute_fisher(
            self.model, 
            self.tokenizer, 
            dsname, 
            reduction=reduction, 
            n_samples=n_samples,
            clip_percentile=fisher_clip_percentile,
            clip_samples=fisher_clip_samples
            )
        elif method == "magnitude":
            self.sensitivity_map = compute_magnitude(self.model)
        elif method == "perturbation":
            self.sensitivity_map = compute_perturbation_sensitivity(
            self.model, 
            self.tokenizer, 
            dsname, 
            n_samples=n_samples
            )
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

    def select_layers_block_aware(self, sensitivity_map, keep_ratio=0.3, block_method="transformer_blocks", verbose=True):
        """
        Block-aware layer selection to minimize format conversions.
        
        Args:
            sensitivity_map: Layer-wise sensitivity scores
            keep_ratio: Fraction of blocks to keep in FP16
            block_method: Method for identifying blocks ('transformer_blocks', 'weakly_connected', 'depth_based')
            verbose: Print diagnostic information
        
        Returns:
            List of layer names to keep in FP16
        """
        from src.selection.block_analysis import BlockAnalyzer
        
        if verbose:
            print(f"\n[BLOCK-AWARE SELECTION] Starting analysis...")
            print(f"[PARAMETERS] keep_ratio={keep_ratio}, method={block_method}")
        
        analyzer = BlockAnalyzer(self.model)
        analyzer.identify_blocks(method=block_method)
        
        if verbose:
            analyzer.visualize_blocks(sensitivity_map)
        
        layers_to_keep = analyzer.select_blocks_to_keep(sensitivity_map, keep_ratio)
        
        if verbose:
            # Show block composition
            block_scores = analyzer.get_block_sensitivity(sensitivity_map)
            sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n[BLOCK SENSITIVITY RANKING]")
            for block_id, score in sorted_blocks[:10]:  # Top 10 blocks
                block_layers = analyzer.blocks[block_id]
                kept = "[KEPT]" if any(layer in layers_to_keep for layer in block_layers) else "[QUANT]"
                print(f"  Block {block_id:2d}: {score:.6f} ({len(block_layers):2d} layers) {kept}")
        
        return list(layers_to_keep)

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
                 sensitivity_ratio=0.05, budget=0.95, verbose=True, invert_selection=False,
                 fisher_clip_percentile=99.0, fisher_clip_samples=32, block_method="transformer_blocks"):

        # Compute sensitivity if not already computed
        if self.sensitivity_map is None:
            if verbose:
                print(f"[COMPUTING SENSITIVITY] Method: {sensitivity_method}")
            self.compute_sensitivity(
                sensitivity_method,
                dsname,
                n_samples,
                fisher_clip_percentile=fisher_clip_percentile,
                fisher_clip_samples=fisher_clip_samples
            )

        selection_method = selection_method.lower()

        # Identify Linear layers
        linear_layer_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layer_names.add(name)
        
        if verbose:
            print(f"[DETECTED] {len(linear_layer_names)} Linear layers in model")
        
        # Filter sensitivity map to only Linear layers
        linear_sensitivity_map = {
            name: score for name, score in self.sensitivity_map.items()
            if name in linear_layer_names
        }
        
        if verbose:
            print(f"[FILTERED] {len(linear_sensitivity_map)} Linear layers have sensitivity scores")
            if len(linear_sensitivity_map) < len(linear_layer_names):
                missing = len(linear_layer_names) - len(linear_sensitivity_map)
                print(f"[WARNING] {missing} Linear layers missing sensitivity scores (will be quantized)")

        # Select layers to keep in high precision
        if selection_method == "knapsack":
            if verbose:
                print(f"[SELECTION METHOD] Knapsack with budget {budget_mb}MB")
            layers_to_keep = knapsack_solver(self.model, linear_sensitivity_map, budget_mb)
        
        elif selection_method == "block_aware":
            if verbose:
                print(f"[SELECTION METHOD] Block-aware with keep_ratio {percentile}")
            layers_to_keep = self.select_layers_block_aware(
                linear_sensitivity_map,
                keep_ratio=percentile,
                block_method=block_method,
                verbose=verbose
            )
        
        else:
            # Compute threshold using ONLY Linear layer scores
            if selection_method == "pct":
                threshold = SelectiveQuantizer.get_threshold_pct(linear_sensitivity_map, percentile)
            elif selection_method == "otsu":
                threshold = SelectiveQuantizer.get_threshold_otsu(linear_sensitivity_map)
            elif selection_method == "elb":
                threshold = SelectiveQuantizer.get_threshold_elb(linear_sensitivity_map)
            elif selection_method == "gradient":
                threshold = SelectiveQuantizer.find_optimal_threshold(linear_sensitivity_map, sensitivity_ratio)
            elif selection_method == "cumulative":
                threshold = SelectiveQuantizer.cumulative_budget_threshold(linear_sensitivity_map, budget)
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")

            if verbose:
                print(f"[THRESHOLD COMPUTED] Method: {selection_method.upper()}, Value: {threshold:.6f}")

            # Apply threshold
            if invert_selection:
                if verbose:
                    print("[INVERTED SELECTION] Keeping LOW sensitivity layers in FP16")
                layers_to_keep = [
                    name for name, score in linear_sensitivity_map.items() 
                    if score <= threshold
                ]
            else:
                if verbose:
                    print("[NORMAL SELECTION] Keeping HIGH sensitivity layers in FP16")
                layers_to_keep = [
                    name for name, score in linear_sensitivity_map.items() 
                    if score >= threshold
                ]

        if verbose:
            print(f"[KEEPING {len(layers_to_keep)} LAYERS IN HIGH PRECISION]")
            
            # DEBUG: Show top sensitive LINEAR layers and which are kept
            print(f"\n[DEBUG] Top 20 most sensitive LINEAR layers:")
            sorted_linear_layers = sorted(linear_sensitivity_map.items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_linear_layers[:20]:
                kept = "[KEPT]" if name in layers_to_keep else "[QUANT]"
                print(f"  {name[:60]:60s} {score:.6f} {kept}")
            
            # Also show BOTTOM 20 (least sensitive)
            print(f"\n[DEBUG] Bottom 20 least sensitive LINEAR layers:")
            for name, score in sorted_linear_layers[-20:]:
                kept = "[KEPT]" if name in layers_to_keep else "[QUANT]"
                print(f"  {name[:60]:60s} {score:.6f} {kept}")

        # Collect layers to quantize
        layers_to_quantize = []
        total_linear = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                total_linear += 1
                if name not in layers_to_keep:
                    layers_to_quantize.append(name)

        if verbose:
            print(f"\n[QUANTIZING {len(layers_to_quantize)}/{total_linear} LAYERS TO INT4]")

        # Handle Accelerate dispatch if present
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

        # Perform quantization with consistent device placement
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

        # Final device consolidation - be careful not to affect kept layers
        if is_dispatched and target_device is not None:
            if verbose:
                print(f"[FINAL CONSOLIDATION] Ensuring all parameters on {target_device}...")
            try:
                # Move only parameters/buffers individually to avoid affecting layer types
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.device != target_device:
                            param.data = param.data.to(target_device)
                    
                    for name, buffer in self.model.named_buffers():
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
        
        # Verify no hooks remain and do final cleanup
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

        # Set model to eval mode
        self.model.eval()

        # Optional validation
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

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Verification: Check what actually happened
        if verbose:
            print("\n[VERIFICATION] Checking actual layer types after quantization:")
            fp16_actual = 0
            int4_actual = 0
            wrong_type = []
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and not isinstance(module, bnb.nn.Linear4bit):
                    # This is FP16 Linear
                    if name in layers_to_keep:
                        fp16_actual += 1
                    else:
                        wrong_type.append(f"  [WARN]  {name} was supposed to be quantized but is still Linear")
                elif isinstance(module, bnb.nn.Linear4bit):
                    # This is INT4
                    int4_actual += 1
                    if name in layers_to_keep:
                        wrong_type.append(f"  [WARN]  {name} was supposed to be kept FP16 but is Linear4bit")
            
            if wrong_type:
                print("\n[WARNING] Type mismatches detected:")
                for msg in wrong_type:
                    print(msg)
            
            print(f"\n[ACTUAL COUNTS] FP16 Linear: {fp16_actual}, INT4 Linear4bit: {int4_actual}")
            print(f"[EXPECTED COUNTS] FP16: {len(layers_to_keep)}, INT4: {len(layers_to_quantize)}")
            
            if fp16_actual == len(layers_to_keep) and int4_actual == len(layers_to_quantize):
                print("[SUCCESS] All layers are correct types!")
            else:
                print("[WARNING] Layer type mismatch detected!")

        # Final report and return
        unchanged_count = total_linear - len(layers_to_quantize)
        compression_ratio = len(layers_to_quantize) / total_linear if total_linear > 0 else 0
        
        if verbose:
            print(f"\n[QUANTIZATION COMPLETE]")
            print(f"  - High precision layers: {unchanged_count}/{total_linear}")
            print(f"  - Quantized layers: {len(layers_to_quantize)}/{total_linear}")
            print(f"  - Compression ratio: {compression_ratio:.1%}")
        
        # Get the device the model is on
        model_device = next(self.model.parameters()).device
        
        if verbose:
            print(f"\n[IMPORTANT] Model is on device: {model_device}")
            print(f"[IMPORTANT] Ensure all inputs during generation are sent to {model_device}")
            print(f"[EXAMPLE] input_ids = input_ids.to('{model_device}')")
        
        return self.model
    
    def get_model_device(self):
        """Helper method to get the device the model is on."""
        return next(self.model.parameters()).device
