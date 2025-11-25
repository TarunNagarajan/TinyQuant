import torch
from tqdm import tqdm
from src.config import Config
from src.data.loader import get_calibration_data

# runs forward/backward pass and accumulates squared gradients
def compute_fisher(model, tokenizer, dsname):
    model.eval() 
    sensitivity_map = {}
    raw_samples = get_calibration_data(dsname, tokenizer)
    
    print(f"[{dsname}] [COMPUTING FISHER INFO ON {Config.DEVICE}]")
    
    for text in tqdm(raw_samples):
        inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True, max_length = 2048).to(Config.DEVICE)
        model.zero_grad()
        outputs = model(**inputs, labels = inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and "weight" in name:
                    grad_sq = param.grad.detach().float().pow(2).sum().item()
                    sensitivity_map[name] = sensitivity_map.get(name, 0.0) + grad_sq

        model.zero_grad()
        del inputs, outputs, loss
        torch.cuda.empty_cache()

    for name in sensitivity_map:
        sensitivity_map[name] /= len(raw_samples) 

    return sensitivity_map    

