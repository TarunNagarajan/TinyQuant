import torch
import torch.nn as nn
import bitsandbytes as bnb 
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator 

class SelectiveQuantizer:
    def __init__(self, model, sensitivity_map):
        self.model = model
        self.map = sensitivity_map

    def _get_threshold_pct(self, percentile):
        scores = list(self.map.values())
        return np.quantile(scores, 1 - percentile)

    def _get_threshold_kms(self):
        # returns the minimum score of the highest cluster
        scores = np.array(list(self.map.values())).reshape(-1, 1)

        kms = KMeans(n_clusters = 2, random_state = 42, n_init = 10)
        kms.fit(scores)

        centers = kms.cluster_centers_
        high_sensitivity_cluster = np.argmax(centers)

        labels = kms.labels_
        high_sensitivity_scores = scores[labels == high_sensitivity_cluster]

        return np.min(high_sensitivity_scores)

    def _get_threshold_elb(self):
        # kneedle algorithm to find the point of maximum curvature
        scores = sorted(list(self.map.values()), reverse = True)
        x = range(len(scores))

        kneedle = KneeLocator(x, scores, curve = "convex", direction = "decreasing")
        index = kneedle.kneedle

        if index is None:
            return scores[int(len(scores) * 0.20)]

        return scores[index]

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

    def quantize(self, method = "kms", percentile = 0.20, verbose = True):
        # main entry point to apply quantization
        method = method.upper()
        if method == "PCT":
            threshold = self._get_threshold_pct(percentile)
        elif method == "KMS":
            threshold = self._get_threshold_kms()
        elif method == "ELB":
            threshold = self._get_threshold_elb()
        else:
            raise ValueError(f"[UNKNOWN METHOD] [{method}]")

        if verbose:
            print(f"[COMPUTED THRESHOLD] [{method}]")

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
                        # low sensitivity, so we'll quantize those
                        replaces.append((name, module))
                    
                    else:
                        unchanged_count += 1 

                else:
                    # missing from the map, defaults to unchanged 
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




