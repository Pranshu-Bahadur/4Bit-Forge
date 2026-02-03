from typing import Dict, List, Tuple, Optional
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(
    dataset_name_or_path : str,
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset(dataset_name_or_path, split="train")
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))

    def preprocess(example):
        messages = [
            {"role": "user", "content": example["instruction"]}, 
            #{"role": "assistant", "content":  example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding='max_length', 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def prepare_embeddings(embed, dataset, X, N, batch_size, device, offload_device):
    bs = batch_size
    B = batch_size
    with torch.no_grad():
        for s in range(0, N, B):
            e = min(N, s + bs)
            ids = torch.cat(dataset[s:e], dim=0).to(device, non_blocking=True)
            x = embed(ids)  # (B,T,H)
            if x.dim() != 3:
                raise RuntimeError(f"embed produced unexpected shape: {tuple(x.shape)}")

            x = x.to(offload_device) if offload_device is not None else x
            B = x.size(0)
            for j in range(B):
                X[s + j] = x[j:j+1].to(offload_device).contiguous()
            del ids, x
    return X