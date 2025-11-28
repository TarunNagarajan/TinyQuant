import torch
import torch.nn as nn
import bitsandbytes as bnb 
import numpy as np
from kneed import KneeLocator
from skimage.filters import threshold_otsu

class SelectiveQuantizer:
    def __init__(self, model, sensitivity_map):
        self.model = model
        self.map = sensitivity_map

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
        # physically replace torch.nn.Linear with bnb.nn.Linear4bit
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = self.model.get_submodule(parent_name)

        new_layer = bnb.nn.Linear4bit(
            input_features = og_layer.in_features,
            output_features = og_layer.out_features,
            compute_dtype = torch.bfloat16,
            bias = og_layer.bias is not None,
            quant_type = "nf4",
        )

        new_layer.weight.data = og_layer.weight.data

        if og_layer.bias is not None:
            new_layer.bias = og_layer.bias 

        new_layer = new_layer.to(og_layer.weight.device)
        setattr(parent, child_name, new_layer)

        del og_layer

    def quantize(self, method="otsu", percentile=0.20, sensitivity_ratio=0.05, budget=0.95, verbose=True):
        method = method.lower()
        
        if method == "pct":
            threshold = SelectiveQuantizer.get_threshold_pct(self.map, percentile)
        elif method == "otsu":
            threshold = SelectiveQuantizer.get_threshold_otsu(self.map)
        elif method == "elb":
            threshold = SelectiveQuantizer.get_threshold_elb(self.map)
        elif method == "gradient":
            threshold = SelectiveQuantizer.find_optimal_threshold(self.map, sensitivity_ratio)
        elif method == "cumulative":
            threshold = SelectiveQuantizer.cumulative_budget_threshold(self.map, budget)
        else:
            raise ValueError(f"[UNKNOWN METHOD] [{method}]")

        if verbose:
            print(f"[COMPUTED THRESHOLD] [{method.upper()}]")

        replaces = []
        unchanged_count = 0
        total_linear = 0
        
        for name, module in self.model.named_modules(): 
            if isinstance(module, nn.Linear):
                total_linear += 1 
                param_key = f"{name}.weight"

                if param_key in self.map:
                    score = self.map[param_key]

                    if score < threshold:
                        replaces.append((name, module))
                    
                    else:
                        unchanged_count += 1 

                else:
                    if verbose:
                        print(f"[MISSING SCORE] [{name}]")

                    unchanged_count += 1 

        if verbose:
            print(f"[QUANTIZING {len(replaces)} LAYERS TO INT4]")
        
        for name, module in replaces:
            self._replace_linear_with_bnb(name, module)

        torch.cuda.empty_cache()

        if verbose:
            print(f"[UNCHANGED: {unchanged_count}/{total_linear} REMAIN IN HIGH PRECISION]")
            print(f"[COMPRESSION RATIO (LAYERS): {len(replaces)/total_linear:.1%}]")

        return self.model




