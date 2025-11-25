import random
from datasets import load_dataset
from src.config import Config

QWEN_MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def format_gsm8k_chat(sample, tokenizer):
   messages = [
           {"role": "system", "content": QWEN_MATH_SYSTEM_PROMPT},
           {"role": "user", "content": sample['question']},
           {"role": "assistant", "content": sample['answer']}
           ]

   return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)

def get_calibration_data(dataset_name, tokenizer, n_samples = None):
    if n_samples is None:
        n_samples = Config.CALIBRATION_SAMPLES

    print(f"[{dataset_name}] | loading {n_samples} samples")

    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split = "train")
        ds = ds.shuffle(seed = 42).select(range(n_samples))
        data = [format_gsm8k_chat(row, tokenizer) for row in ds]

    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split = "train")
        cleantext = [x['text'] for x in ds if len(x['text']) > 100 and not x['text'].startswith(" =")]

        random.seed(42)

        if (len(cleantext) < n_samples):
            data = cleantext
        else:
            data = random.sample(cleantext, n_samples)

    else:
        raise ValueError("unknown dataset.")

    print(f"--- [DEBUG] Sample 0 ({dataset_name}) ---")
    print(data[0][:300].replace('\n', '\\n'))
    print("------------------------------------------")

    return data

def get_eval_dataset():
    return load_dataset("gsm8k", "main", split = "test")



