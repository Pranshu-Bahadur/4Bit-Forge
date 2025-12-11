import os
import json
from warnings import warn
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, DeepseekV3ForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3MLP
from safetensors.torch import save_file


@torch.no_grad()
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
    # Assertions
    assert backend_layout in ('vllm_gptq', 'triton_gptq'), "Unsupported backend layout."
    assert isinstance(model, AutoModelForCausalLM), "Model must be an instance of AutoModelForCausalLM."
    assert hasattr(model, 'get_input_embeddings'), "Model must have 'get_input_embeddings' method."
    assert hasattr(model, 'model'), "Model must have a 'model' attribute."
    assert hasattr(model.model, 'layers'), "Model's 'model' must have a 'layers' attribute."
    for layer_attr in ('input_layernorm', 'self_attn', 'post_attention_layernorm', 'mlp'):
        assert hasattr(model.model.layers[0], f'{layer_attr}'), f"Model layers must have '{layer_attr}' attribute."
    
    # Setting model to eval mode
    model = model.eval()

    # Getting input embeddings
    n_calib_tokens = 0
    data_idx = 0
    input_embeddings = []
    while n_calib_tokens < num_calibration_tokens:
        sample = dataset[data_idx]['input_ids'].to(model.devaice)
        embeddings = model.get_input_embeddings()(sample)
        input_embeddings.append(embeddings)
        n_calib_tokens += embeddings.shape[0]
        data_idx += 1
    
    if n_calib_tokens < num_calibration_tokens:
        warn(f"Only {n_calib_tokens} tokens available for calibration, less than requested {num_calibration_tokens}.")
    print(f"Collected {n_calib_tokens} tokens for calibration with {data_idx} data samples.")

    # Main loop down the model layers
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    model_index = _dispatch_quantize_hf_model(
        model,
        input_embeddings,
        bits,
        group_size,
        backend_layout,
        layers_to_quantize,
        output_checkpoint_dir
    )
        
    # Storing model index
    model_index_path = os.path.join(output_checkpoint_dir, 'model.safetensors.index.json')
    with open(model_index_path, 'w') as f:
        json.dump(model_index, f, indent=2)

        
        
def _dispatch_quantize_hf_model(
    model: AutoModelForCausalLM,
    input_embeddings: list[torch.Tensor],
    bits: int,
    group_size: int,
    backend_layout: str,
    layers_to_quantize: Iterable[str],
    output_checkpoint_dir: str,
):
    if isinstance(model, DeepseekV3ForCausalLM):
        fn = _quantize_deepseek_model
    else:
        raise NotImplementedError(f"Quantization for model type {type(model)} is not implemented.")
    return fn(
        model,
        input_embeddings,
        bits,
        group_size,
        backend_layout,
        layers_to_quantize,
        output_checkpoint_dir
    )
        
@torch.no_grad()
def _quantize_deepseek_model(
    model: DeepseekV3ForCausalLM,
    input_embeddings: list[torch.Tensor],
    bits: int,
    group_size: int,
    backend_layout: str,
    layers_to_quantize: Iterable[str],
    output_checkpoint_dir: str,
):
    # TODO: Currently, this methods only quantizes routed experts
    # In the future, we should consider the `layers_to_quantize` argument and quantize other layers as well.
    
    model_index = {}
    position_ids = [
        torch.arange(embeds.size(0), device=embeds.device).unsqueeze(0)
        for embeds in input_embeddings
    ]
    pos_embeds = [
        model.rotary_emb(embeds, position_ids=position_ids)
        for embeds in input_embeddings
    ]
    hidden_states_list = input_embeddings
    for layer_idx in range(len(model.model.layers)):
        for hidden_states in hidden_states_list:
            # Self-attention: TODO - currently not quantizing attention layers
            residual = hidden_states
            hidden_states = model.model.layers[layer_idx].self_attn.input_layernorm(hidden_states)
            hidden_states, _ = model.model.layers[layer_idx].self_attn(
                hidden_states=hidden_states,
                position_embeds=pos_embeds,
            )
            hidden_states = hidden_states + residual
            
            # MLP
            residual = hidden_states
            hidden_states = model.model.layers[layer_idx].post_attention_layernorm(hidden_states)
            if isinstance(model.model.layers[layer_idx].mlp, DeepseekV3MLP):
                # MLP without experts is skipped: TODO - currently not quantizing dense MLP layers
                hidden_states = model.model.layers[layer_idx].mlp(hidden_states)                
            elif isinstance(model.model.layers[layer_idx].mlp, DeepseekV3MoE):
                # Skipping quantization of router
                residual_inner = hidden_states
                orig_shape = hidden_states.shape
                router_logits = model.model.layers[layer_idx].mlp.gate(hidden_states)
                topk_indices, topk_weights = model.model.layers[layer_idx].mlp.route_tokens_to_experts(router_logits)
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) # Flattening tokens
                
                # Quantizing experts
                experts_out = _quantize_deepseek_experts(
                    model,
                    layer_idx,
                    hidden_states,
                    topk_indices,
                    topk_weights,
                    bits,
                    group_size,
                    backend_layout,
                    output_checkpoint_dir
                )

                hidden_states += experts_out.view(orig_shape)
                
                # Skipping quantization of shared expert
                hidden_states += model.model.layers[layer_idx].mlp.shared_experts(residual_inner)
            else:
                raise ValueError(f"Unknown MLP type {type(model.model.layers[layer_idx].mlp)} in DeepseekV3 model.")

            hidden_states = hidden_states + residual
    
    return model_index

    
@torch.no_grad()
def _quantize_deepseek_experts(
    model: DeepseekV3ForCausalLM,
    layer_idx: int,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    bits: int,
    group_size: int,
    backend_layout: str,
    output_checkpoint_dir: str,
):
    """Quantizes and stores experts while returning their new output."""
    # TODO
    ...

    """
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=model.model.layers[layer_idx].mlp.num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == model.model.layers[layer_idx].mlp.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, model.model.layers[layer_idx].mlp.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = model.model.layers[layer_idx].mlp.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, model.model.layers[layer_idx].mlp.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * topk_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states
    """