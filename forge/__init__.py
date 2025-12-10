from typing import Iterable
import torch

def gptq_quantize(
    model: torch.nn.Module,
    dataset: Iterable,
    bits: int = 4,
    group_size: int = 128,
    num_calibration_tokens: int = 1000,
    backend_layout: str = 'vllm_gptq',
    layers_to_quantize: Iterable[str] = ('*experts*',),
    output_checkpoint_dir: str = 'quantized_model',
):
    # TODO
    ...