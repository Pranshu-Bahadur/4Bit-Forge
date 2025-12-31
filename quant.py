import os
import gc
import re
import argparse
from tqdm import tqdm

import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from accelerate import init_empty_weights

import forge.utils

from forge.gptq import GPTQ

ROUTED_EXPERTS_REGEX = r".*\.experts\.\d+\.(down|gate|up|gate_up|w\d+)_proj$"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument("--num_calibration_samples", default=512, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)

    parser.add_argument("--calib_batch_size", type=int, default=8)

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4],
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
    )

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["sdpa", "eager", "flash_attention_2"])
    
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--rel_damp", type=float, default=1e-1)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--quantization_scale", type=str, default="mse", choices=["absmax", "mse"])
    parser.add_argument("--quantization_order", type=str, default="activation", choices=["default", "activation"])
    parser.add_argument(
        "--quantize_only_routed_experts",
        default=False,
        action="store_true",
    )
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--offload", action="store_true")

    parser.add_argument("--owner_gptq_handles", action="store_true", help="Reuse hessians between gate and up projections.")
    
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--dtype", default="bfloat16", type=str, choices=["float16", "bfloat16"])
    

    parser.add_argument("--jit_stream", action="store_true",
                   help="Enable Ghost Loader + JIT shard streaming (requires safetensors index).")
    parser.add_argument("--hf_tmp_dir", type=str, default="/tmp/hf_jit",
                   help="Temporary dir used to download shards (disk window kept small).")
    parser.add_argument("--lru_ram_gb", type=float, default=0.0,
                   help="Optional RAM-side shard cache size (GB). 0 disables caching.")
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    rank = 0 #TODO multigpu
    device = "cuda"
    torch.set_grad_enabled(False)
    #torch.cuda.set_device(device)
    dtype = getattr(torch, str(args.dtype))

    os.environ.setdefault("HF_HOME", args.hf_tmp_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_tmp_dir)
    os.environ.setdefault("HF_HUB_CACHE", args.hf_tmp_dir)

    offload_device = "cpu" if args.offload else None

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config, trust_remote_code=True, 
            attn_implementation=str(args.attn_implementation), 
            torch_dtype=dtype
        ).eval()
        model.config.use_cache = False
    ROUTED_EXPERTS_REGEX = r".*mlp\.experts\.\d+\.(down|gate|up|gate_up)_proj$"
    #ROUTED_EXPERTS_REGEX = ROUTED_EXPERTS_REGEX if "deepseek" in str(args.model_name_or_path).lower() else r".*mlp\.experts\.(down|gate|up|gate_up)_proj.*"
    shard_ids = forge.utils.io.load_safetensors_index(args.model_name_or_path, tmp_dir=args.hf_tmp_dir)
    weight_map = shard_ids["weight_map"]
    num_shards = len(set(weight_map.values()))


    # --- IO objects (create once) ---
    lru = forge.utils.io.ShardLRU(max_bytes=int(args.lru_ram_gb * (1024**3)))

    assumed_shard_bytes = int(4.63 * (1024**3)) if args.jit_stream else 0

    disk_cfg = forge.utils.io.DiskWindowConfig(
        shard_bytes=assumed_shard_bytes,
        safety_bytes=int(2 * (1024**3)),
        max_shards=8,
        use_lru_touch=True,
    )
    disk_window = forge.utils.io.ShardDiskWindow(args.hf_tmp_dir, disk_cfg)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
                print("[WARN] Tokenizer has no pad_token_id! Using eos_token_id as pad_token.")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

    #Assumptions: your model has a chat template & your dataset contains instruction & output columns & contains split=train
    calibration_dataset = forge.utils.preprocess.prepare_dataset(str(args.dataset_name_or_path),
                                                                tokenizer, 
                                                                int(args.max_sequence_length), 
                                                                int(args.num_calibration_samples))

    del tokenizer
    gc.collect()

    N = len(calibration_dataset)
    T = int(calibration_dataset[0].shape[-1])
    B = max(1, int(args.calib_batch_size))

    X = [None] * N

    embed = model.model.embed_tokens

    forge.utils.io.jit_load_prefix_to_cpu(
        model,
        args.model_name_or_path,
        weight_map,
        ["model.embed_tokens."],
        args.hf_tmp_dir,
        lru,
        reserve_bytes=0,
        disk_window=disk_window,
    )


    embed.to(device)
    X = forge.utils.preprocess.prepare_embeddings(embed, calibration_dataset, X, N, B, device, offload_device)
    if args.offload:
        embed.to("cpu")
        forge.utils.io.metaize_module_(embed)

    if offload_device == "cpu":
        X = [t.to("cpu", non_blocking=True) for t in X]
        torch.cuda.empty_cache()
    position_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

    rotary_emb = getattr(model.model, "rotary_emb", None)

    if rotary_emb:
        forge.utils.io.jit_load_prefix_to_cpu(
            model,
            args.model_name_or_path,
            weight_map,
            ["model.rotary_emb."],
            args.hf_tmp_dir,
            lru,
            reserve_bytes=0,
            disk_window=disk_window,
        )
        
        rotary_emb = forge.utils.engine.ensure_rotary_emb(
            model, config,
            device=device
        )
    else:
        rotary_emb = forge.utils.engine.find_rotary_emb(model)
        forge.utils.io.jit_load_prefix_to_cpu(
            model,
            args.model_name_or_path,
            weight_map,
            ["model.rotary_emb."],
            args.hf_tmp_dir,
            lru,
            reserve_bytes=0,
            disk_window=disk_window,
        )
        rotary_emb = forge.utils.engine.ensure_rotary_emb(
            model, config,
            device=device
        )
    

    blocks = model.model.layers

    for block_id, block in tqdm(enumerate(blocks)):
        prefix = f"model.layers.{block_id}."
        forge.utils.io.jit_load_prefix_to_cpu(
            model,
            args.model_name_or_path,
            weight_map,
            [prefix],
            args.hf_tmp_dir,
            lru,
            reserve_bytes=0,
            disk_window=disk_window,
            materialize_block=block,
            materialize_prefix=prefix,
            materialize_fn=forge.utils.io.materialize_block_weights_to_fp,
            group_size=int(args.group_size),
            bits=int(args.bits),
            dtype=dtype,
            list_layers_fn=forge.utils.engine.list_layers,
        )
        meta_names = [n for n, p in block.named_parameters(recurse=True) if getattr(p, "is_meta", False)]
        assert not meta_names, f"Still meta params in block {block_id}: {meta_names[:10]}"
        block.to(device)
        
        #Calibration Pass
        handles = {}
        hooks  = {}
        layers = forge.utils.engine.list_layers(block) #TODO sort

        if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
            experts = block.mlp.experts
            if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
                
                for layer_name, layer in layers.items():
                    if args.quantize_only_routed_experts and re.search(ROUTED_EXPERTS_REGEX, layer_name) is None:
                        continue
                    owner = None

                    handles[layer_name] = GPTQ(
                            layer=layer,
                            group_size=args.group_size,
                            symmetric=bool(args.sym),
                            rel_damp=args.rel_damp,
                            quantization_order=args.quantization_order,
                            quantization_scale=args.quantization_scale,
                            owner=owner,
                            algorithm="gptq",
                            device = device
                        )
                experts_hook = forge.utils.engine.fused_expert_hooks(block, handles)
                hooks['fused_experts'] = experts.register_forward_hook(experts_hook, with_kwargs=True)
                
        else:

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    x = inp[0] if isinstance(inp, (tuple, list)) else inp
                    handles[name].update(x)                
                return _hook
        
            for layer_name, layer in layers.items():
                if args.quantize_only_routed_experts and re.search(ROUTED_EXPERTS_REGEX, layer_name) is None:
                    continue
                owner = None
                if args.owner_gptq_handles and layer_name.endswith("up_proj"):
                    parent_name, _ = layer_name.rsplit(".", 1)
                    owner = handles.get(f"{parent_name}.gate_proj", None)

                handles[layer_name] = GPTQ(
                        layer=layer,
                        group_size=args.group_size,
                        symmetric=bool(args.sym),
                        rel_damp=args.rel_damp,
                        quantization_order=args.quantization_order,
                        quantization_scale=args.quantization_scale,
                        owner=owner,
                        algorithm="gptq",
                        device = device
                    )
                if owner is None:
                    hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))


        with torch.no_grad():
            _ = forge.utils.engine.forward(block, X, position_ids, N, B, device, offload_device, False, rotary_emb)
            
            for _, h in hooks.items():
                h.remove()

            for handle_name, handle in handles.items():
                if handle._is_owner():
                    qweight, scales, qzeros = handle.quantize()
                    deq, scales, qzeros = forge.utils.engine.dequantize_forge_full(handle.layer.weight.dtype, qweight, scales, qzeros,
                                                                int(args.group_size), int(args.bits),
                                                                )
                    handle.layer.weight.copy_(deq)
                    if args.save_dir:
                        os.makedirs(os.path.join(args.save_dir + f"/block.{block_id}", handle_name), exist_ok=True)
                        torch.save(
                                {"qweight": qweight.to(torch.int8), "scale": scales.to(dtype), "zero": qzeros.to(torch.int8)},
                                os.path.join(args.save_dir, f"block.{block_id}", handle_name, f"quantized_weight.pt"),
                            )

            for handle_name, handle in handles.items():
                if not handle._is_owner():
                    qweight, scales, qzeros = handle.quantize()
                    deq, scales, qzeros = forge.utils.engine.dequantize_forge_full(handle.layer.weight.dtype, qweight, scales, qzeros,
                                                                int(args.group_size), int(args.bits),
                                                                )
                    handle.layer.weight.copy_(deq)
                    if args.save_dir:
                        os.makedirs(os.path.join(args.save_dir + f"/block.{block_id}", handle_name), exist_ok=True)
                        torch.save(
                                {"qweight": qweight.to(torch.int8), "scale": scales.to(dtype), "zero": qzeros.to(torch.int8)},
                                os.path.join(args.save_dir + f"/block.{block_id}", handle_name, f"quantized_weight.pt"),
                            )

            for _, handle in handles.items():
                handle.reset()

            del handles
            del hooks
            del deq
            torch.cuda.empty_cache()
            gc.collect()

        #Activation Update
        with torch.inference_mode():
            X = forge.utils.engine.forward(block, X, position_ids, N, B, device, offload_device, True, rotary_emb)

        if args.offload:
            block.to("cpu")
            forge.utils.io.metaize_module_(block)

        torch.cuda.empty_cache()
        gc.collect()

    if args.save_dir:
        torch.save(
            {
                "bits": int(args.bits),
                "group_size": int(args.group_size),
                "quantize_only_experts": bool(args.quantize_only_experts)
            },
            os.path.join(args.save_dir, "metadata.pt")
        )

if __name__ == "__main__":
    main()