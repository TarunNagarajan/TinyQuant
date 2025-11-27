import random
from datasets import load_dataset
from src.config import Config

def get_calibration_data(dataset_name, n_samples=None):
    if n_samples is None:
        n_samples = Config.CALIBRATION_SAMPLES

    print(f"[{dataset_name}] | [LOADING {n_samples} SAMPLES]")

    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        ds = ds.shuffle(seed=42).select(range(n_samples))
        data = [row['question'] for row in ds]

    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cleantext = [x['text'] for x in ds if len(x['text']) > 100 and not x['text'].startswith(" =")]

        random.seed(42)

        if len(cleantext) < n_samples:
            data = cleantext
        else:
            data = random.sample(cleantext, n_samples)
    
    elif dataset_name == "math":
        ds = load_dataset("lighteval/MATH", split="train")
        ds = ds.shuffle(seed=42).select(range(n_samples))
        data = [row['problem'] for row in ds]

    else:
        raise ValueError(f"Unknown dataset for calibration: {dataset_name}")

    return data

def get_eval_dataset(dataset_name="gsm8k"):
    if dataset_name == "gsm8k":
        return load_dataset("gsm8k", "main", split="test")
    elif dataset_name == "math":
        return load_dataset("lighteval/MATH", split="test")
    else:
        raise ValueError(f"Evaluation dataset for '{dataset_name}' not configured.")
