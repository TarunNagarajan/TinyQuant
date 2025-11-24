import torch
import os
import sys

class Config:
    DEVICE = "cuda" if torch.cuda_is_available() else "cpu"
    DTYPE = torch.bfloat16
    MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    CALIBRATION_SAMPLES = 64
    CALIBRATION_BATCH_SIZE = 1 
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
    LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

    def savedirs():
        os.makedirs(Config.MAPS_DIR, exist_ok = True)
        os.makedirs(Config.LOGS_DIR, exist_ok = True)

Config.savedirs()
