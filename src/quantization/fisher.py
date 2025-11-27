import torch
from tqdm import tqdm
from src.config import Config
from src.data.loader import get_calibration_data

def compute_fisher(model, tokenizer, dsname, reduction="mean", n_samples=None):
    assert reduction in ("mean", "sum")

    model.eval()
    sensitivity_map = {}
    raw_samples = get_calibration_data(dsname, n_samples=n_samples)

    print(f"[{dsname}] [COMPUTING FISHER INFO]")

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

        if "labels" in inputs and inputs["labels"] is not None:
            active_token_count = int((inputs["labels"] != -100).sum().item())
        else:
            active_token_count = int(inputs["input_ids"].numel())

        loss = outputs.loss * max(active_token_count, 1)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and "weight" in name:
                    grad_sq = param.grad.detach().cpu().float().square().mean().item()
                    sensitivity_map[name] = sensitivity_map.get(name, 0.0) + grad_sq

        model.zero_grad()
        del inputs, outputs, loss
        processed += 1

        if torch.cuda.is_available() and ((i + 1) % 10 == 0):
            torch.cuda.empty_cache()

    if reduction == "mean" and processed > 0:
        for name in sensitivity_map:
            sensitivity_map[name] /= processed

    return sensitivity_map
