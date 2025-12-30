import os
import json

import torch
import torch.nn as nn

from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import hf_hub_download


def load_safetensors_index(repo_id : str, tmp_dir : str) -> dict:
    os.makedirs(tmp_dir, exist_ok = True)
    fpath = hf_hub_download(
        repo_id = repo_id,
        filename = "model.safetensors.index.json",
        repo_type = "model",
        local_dir = tmp_dir,
        force_download = False
    )
    with open(fpath, "r") as f:
        return json.load(f)


def jit_load_prefix_to_cpu(
        model : nn.Module,
        repo_id : str,
        weight_map : dict,
        prefixes : list,
        tmp_dir : str):
    
    needed = [*(set(weight_map.keys()) & set(prefixes))]
    assert len(needed) == len(prefixes)
    pass

