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
        # returns the elbow point after which the sensitivity drops off
        scores = sorted(list(self.map.values()), reverse = True)
        x = range(len(scores))

        kneedle = KneeLocator(x, scores, curve = "convex", direction = "decreasing")
        index = kneedle.kneedle

        if index is None:
            return score[int(len(scores) * 0.20)]

        return scores[index]





