import torch
import os

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    
    CALIBRATION_SAMPLES = 64
    CALIBRATION_BATCH_SIZE = 1
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # These will be set by set_model()
    MODEL_NAME = None
    MODEL_ID = None
    MAPS_DIR = None
    LOGS_DIR = None

    @staticmethod
    def set_model(model_name):
        Config.MODEL_NAME = model_name
        if model_name == "qwen":
            Config.MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        elif model_name == "llama":
            Config.MODEL_ID = "unsloth/llama-3.2-1b-Instruct-hf"
        elif model_name == "phi":
            Config.MODEL_ID = "microsoft/phi-2"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Model-specific output directories
        Config.MAPS_DIR = os.path.join(Config.RESULTS_DIR, "maps", model_name)
        Config.LOGS_DIR = os.path.join(Config.RESULTS_DIR, "logs", model_name)
        
        # Create directories if they don't exist
        os.makedirs(Config.MAPS_DIR, exist_ok=True)
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
