import torch
from tqdm import tqdm
from src.config import Config
from src.data.loader import get_calibration_data

def compute_fisher(model, tokenizer, dsname, reduction="mean", n_samples=None):
    assert reduction in ("mean", "sum")

    model.eval()
    sensitivity_map = {}
    raw_samples = get_calibration_data(dsname, n_samples=n_samples)

    print(f"[{dsname}] [COMPUTING FISHER INFO - DENSITY CORRECTED]")

    processed = 0
    for i, text in enumerate(tqdm(raw_samples, desc="Computing Fisher", leave=False, total=len(raw_samples))):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

        model.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and "weight" in name:
                    # 1. Calculate Sum of Squared Gradients (Raw Sensitivity)
                    grad_sq = param.grad.detach().cpu().float().square().sum().item()
                    
                    # 2. CRITICAL FIX: Normalize by parameter count
                    # This computes "Sensitivity per Parameter" (Value Density)
                    # identifying small but critical layers (like LayerNorms/Attention)
                    density_score = grad_sq / param.numel()
                    
                    sensitivity_map[name] = sensitivity_map.get(name, 0.0) + density_score

        model.zero_grad()
        del inputs, outputs, loss
        processed += 1

        if torch.cuda.is_available() and ((i + 1) % 10 == 0):
            torch.cuda.empty_cache()

    if reduction == "mean" and processed > 0:
        for name in sensitivity_map:
            sensitivity_map[name] /= processed

    return sensitivity_map
