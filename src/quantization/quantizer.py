import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit

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

    def _replace_linear_with_bnb(self, full_name, og_layer):
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
        else:
            parent = self.model
            child_name = full_name

        # 1. Create the 4bit layer on CPU first
        new_layer = bnb.nn.Linear4bit(
            input_features=og_layer.in_features,
            output_features=og_layer.out_features,
            compute_dtype=torch.bfloat16, # Ensure this matches your model dtype
            bias=og_layer.bias is not None,
            quant_type="nf4",
        )

        # 2. CRITICAL FIX: Move weights to CPU and Wrap in Params4bit
        # We must detach to CPU so that the .to(device) call later TRIGGERS the quantization.
        # If we just copy GPU->GPU, bnb often skips the quantization step.
        weight_cpu = og_layer.weight.data.cpu()
        
        new_layer.weight = Params4bit(
            weight_cpu, 
            requires_grad=False, 
            quant_type="nf4"
        )

        # Handle Bias (Bias is not quantized, stays FP16)
        if og_layer.bias is not None:
            new_layer.bias = nn.Parameter(og_layer.bias.data.cpu())

        # 3. Move to Target Device -> THIS IS WHEN COMPRESSION HAPPENS
        target_device = og_layer.weight.device
        new_layer = new_layer.to(target_device)

        # 4. Swap the layer
        setattr(parent, child_name, new_layer)

        # 5. Cleanup
        del og_layer

    def quantize(self, selection_method="knapsack", sensitivity_method="perturbation", dsname="gsm8k", n_samples=32, budget_mb=4096, percentile=0.20, sensitivity_ratio=0.05, budget=0.95, verbose=True):
        
        # 1. Compute sensitivity if not already computed
        if self.sensitivity_map is None:
            self.compute_sensitivity(sensitivity_method, dsname, n_samples)

        selection_method = selection_method.lower()

        if selection_method == "knapsack":
            layers_to_keep = knapsack_solver(self.model, self.sensitivity_map, budget_mb)
        else:
            # Existing threshold-based methods
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
                raise ValueError(f"[UNKNOWN SELECTION METHOD] [{selection_method}]")
            
            if verbose:
                print(f"[COMPUTED THRESHOLD] [{selection_method.upper()}]")
            
            layers_to_keep = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if name in self.sensitivity_map:
                        if self.sensitivity_map[name] >= threshold:
                            layers_to_keep.append(name)
                    else:
                        # If a linear layer has no score, it will not be kept and will be quantized.
                        # This is a reasonable default for layers without sensitivity info.
                        pass

        # Quantize layers that are not in the keep list
        replaces = []
        total_linear = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                total_linear += 1
                if name not in layers_to_keep:
                    replaces.append((name, module))
        
        unchanged_count = total_linear - len(replaces)

        if verbose:
            print(f"[QUANTIZING {len(replaces)} LAYERS TO INT4]")
        
        for name, module in replaces:
            self._replace_linear_with_bnb(name, module)

        torch.cuda.empty_cache()

        if verbose:
            print(f"[UNCHANGED: {unchanged_count}/{total_linear} REMAIN IN HIGH PRECISION]")
            print(f"[COMPRESSION RATIO (LAYERS): {len(replaces)/total_linear:.1%}]")

        return self.model
