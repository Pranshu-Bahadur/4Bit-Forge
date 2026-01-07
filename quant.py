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

def _to_u8_nibble_sym_int4(qweight_int8: torch.Tensor) -> torch.Tensor:
    """
    Convert symmetric int4 stored as int8 in [-8..7] into uint8 nibbles [0..15]
    so that dequant in matmul uses (nibble - 8).
    If your qweight is already 0..15, this still works fine.
    """
    # promote to int16 so +8 doesn't overflow
    qw = qweight_int8.to(torch.int16)
    qw = (qw + 8).clamp_(0, 15).to(torch.uint8)
    return qw.contiguous()

def _parse_expert_id(handle_name: str) -> int:
    m = re.search(r"\.experts\.(\d+)\.", handle_name)
    if not m:
        raise RuntimeError(f"Could not parse expert id from handle_name={handle_name}")
    return int(m.group(1))

def _save_Wpair_u64(path: str, Wpair_u64: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Wpair_u64 is uint64 [G2, R, 2] (contiguous)
    torch.save({"Wpair_u64": Wpair_u64.contiguous()}, path)
    torch.save({"Wpair_u64": Wpair_u64.contiguous()}, path)



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

    parser.add_argument(
        "--algorithm",
         type=str, default="gptq",
                   choices=["gptq", "sparsegptq"])

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["sdpa", "eager", "flash_attention_2"])
    
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--rel_damp", type=float, default=1e-1)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--quantization_scale", type=str, default="mse", choices=["absmax", "mse"])
    parser.add_argument("--quantization_order", type=str, default="default", choices=["default", "activation"])
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
    ROUTED_EXPERTS_REGEX = r".*\.experts\.\d+\.(down_proj|gate_proj|up_proj|gate_up_proj|w\d+)$"

    #ROUTED_EXPERTS_REGEX = ROUTED_EXPERTS_REGEX if "deepseek" in str(args.model_name_or_path).lower() else r".*mlp\.experts\.(down|gate|up|gate_up)_proj.*"
    shard_ids = forge.utils.io.load_safetensors_index(args.model_name_or_path, tmp_dir=args.hf_tmp_dir)
    weight_map = shard_ids["weight_map"]
    num_shards = len(set(weight_map.values()))


    # --- IO objects (create once) ---
    lru = forge.utils.io.ShardLRU(max_bytes=int(args.lru_ram_gb * (1024**3)))

    assumed_shard_bytes = int(5.36 * (1024**3)) if args.jit_stream else 0

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
    if tokenizer.chat_template is None:
          tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if not thinking is defined %}{% set thinking = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, system_prompt='', is_first_sp=true, is_last_user=false, is_only_sys=false, is_prefix=false) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{% set ns.is_only_sys = true %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{%- set ns.is_first = false -%}{%- set ns.is_last_user = true -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}{%- if ns.is_last_user or ns.is_only_sys %}{{'<｜Assistant｜></think>'}}{%- endif %}{%- set ns.is_last_user = false -%}{%- set ns.is_first = false %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}{%- else %}{{message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + '<｜tool▁sep｜>' + tool['function']['arguments'] + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none) %}{%- if ns.is_last_user %}{{'<｜Assistant｜>'}}{%- if message['prefix'] is defined and message['prefix'] and thinking %}{{'<think>'}}{%- else %}{{'</think>'}}{%- endif %}{%- endif %}{%- if message['prefix'] is defined and message['prefix'] %}{%- set ns.is_prefix = true -%}{%- endif %}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{%- set content = message['content'] -%}{%- if '</think>' in content %}{%- set content = content.split('</think>', 1)[1] -%}{%- endif %}{{content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_last_user = false -%}{%- set ns.is_tool = true -%}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- if message['role'] != 'system' %}{% set ns.is_only_sys = false %}{%- endif %}{%- endfor -%}{% if add_generation_prompt and not ns.is_tool%}{% if ns.is_last_user or ns.is_only_sys or not ns.is_prefix %}{{'<｜Assistant｜>'}}{%- if not thinking %}{{'</think>'}}{%- else %}{{'<think>'}}{%- endif %}{% endif %}{% endif %}"

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

        if hasattr(block, "mlp") and hasattr(block.mlp, "experts") and hasattr(block.mlp.experts, "gate_up_proj") and hasattr(block.mlp.experts, "down_proj"):
                experts = block.mlp.experts
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
                            algorithm=str(args.algorithm),
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
        
            layers = dict(sorted(layers.items()))
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
                        algorithm=str(args.algorithm),
                        device = device
                    )
                if owner is None:
                    hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        with torch.no_grad():
            _ = forge.utils.engine.forward(block, X, position_ids, N, B, device, offload_device, False, rotary_emb)
            del _
            
            for _, h in hooks.items():
                h.remove()

            _pending_gate_Wpair = {}

            for handle_name, handle in handles.items():
                if handle._is_owner():
                    # inside your loop:
                    if str(args.algorithm) == "sparsegptq":
                        qweight, scales, qzeros, M = handle.quantize()

                        # keep this if you still want dense deq checkpoint / sanity path
                        deq = forge.utils.engine.dequantize_forge_full(
                            handle.layer.weight.dtype,
                            qweight, scales, qzeros,
                            int(args.group_size), int(args.bits),
                        )
                        handle.layer.weight.copy_(deq)

                        # ---- PACK NOW (no intermediate disk) ----
                        # qweight is [R,C] int8, M is [G32,R] uint32, scales is [G32,R] float-ish
                        # pack kernel expects qweight_rc uint8 nibbles 0..15
                        qw_u8 = _to_u8_nibble_sym_int4(qweight)

                        # ensure dtypes match packer expectations
                        scales_f32 = scales.to(torch.float32).contiguous()
                        M_u32 = M.to(torch.uint32).contiguous()

                        # packed: uint64 [G2, R, 2]
                        G32 = (qw_u8.shape[1] + 32 - 1) // 32
                        Wpair_u64 = forge.backend.cuda.kernels.pack_sparsegptq14_to_u64x2(
                            qw_u8, M_u32, scales_f32.view(qw_u8.shape[0], G32).transpose(0, 1).contiguous()
                        )

                        # ---- SAVE PACKED IN YOUR DESIRED LAYOUT ----
                        if args.save_dir:
                            # Detect proj type from handle_name
                            is_gate = handle_name.endswith("gate_proj")
                            is_up   = handle_name.endswith("up_proj")
                            is_down = handle_name.endswith("down_proj")

                            if is_gate:
                                eid = _parse_expert_id(handle_name)
                                _pending_gate_Wpair[(block_id, eid)] = Wpair_u64.detach().contiguous()

                            elif is_up:
                                eid = _parse_expert_id(handle_name)
                                key = (block_id, eid)
                                if key not in _pending_gate_Wpair:
                                    raise RuntimeError(f"Missing cached gate Wpair for block={block_id} expert={eid}. "
                                                    f"Loop order unexpected; need to pack+cache gate first.")

                                Wgate = _pending_gate_Wpair.pop(key)  # [G2, 2048, 2]
                                Wup   = Wpair_u64.detach().contiguous()  # [G2, 2048, 2]

                                # Stack along R -> [G2, 4096, 2]
                                W13 = torch.cat([Wgate, Wup], dim=1).contiguous()

                                # Save under gate_up_proj folder (Option B)
                                gateup_name = handle_name.replace("up_proj", "gate_up_proj")
                                out_dir = os.path.join(args.save_dir, f"block.{block_id}", gateup_name)
                                _save_Wpair_u64(os.path.join(out_dir, "Wpair_u64.pt"), W13)

                                del Wgate, Wup, W13

                            elif is_down:
                                out_dir = os.path.join(args.save_dir, f"block.{block_id}", handle_name)
                                _save_Wpair_u64(os.path.join(out_dir, "Wpair_u64.pt"), Wpair_u64.detach())

                            else:
                                # If you have non-expert or different names, decide what to do here.
                                pass

                        # aggressively free
                        del deq, qw_u8, scales_f32, M_u32, Wpair_u64
                        del qweight, scales, qzeros, M
                    else:
                        qweight, scales, qzeros = handle.quantize()

                        deq = forge.utils.engine.dequantize_forge_full(
                            handle.layer.weight.dtype,
                            qweight, scales, qzeros,
                            int(args.group_size), int(args.bits),
                        )
                        handle.layer.weight.copy_(deq)

                        if args.save_dir:
                            out_dir = os.path.join(args.save_dir, f"block.{block_id}", handle_name)
                            os.makedirs(out_dir, exist_ok=True)
                            torch.save(
                                {
                                    "qweight": qweight.to(torch.int8),        # pre-nibble
                                    "scales": scales.to(torch.float32),
                                    "qzeros": qzeros.to(torch.int32),
                                    "group_size": int(args.group_size),
                                    "bits": int(args.bits),
                                    "shape": tuple(handle.layer.weight.shape),
                                },
                                os.path.join(out_dir, "quantized_weight.pt"),
                            )
                        del deq, scales, qzeros

            for handle_name, handle in handles.items():
                if not handle._is_owner():
                    # inside your loop:
                    if str(args.algorithm) == "sparsegptq":
                        qweight, scales, qzeros, M = handle.quantize()

                        # keep this if you still want dense deq checkpoint / sanity path
                        deq = forge.utils.engine.dequantize_forge_full(
                            handle.layer.weight.dtype,
                            qweight, scales, qzeros,
                            int(args.group_size), int(args.bits),
                        )
                        handle.layer.weight.copy_(deq)

                        # ---- PACK NOW (no intermediate disk) ----
                        # qweight is [R,C] int8, M is [G32,R] uint32, scales is [G32,R] float-ish
                        # pack kernel expects qweight_rc uint8 nibbles 0..15
                        qw_u8 = _to_u8_nibble_sym_int4(qweight)

                        # ensure dtypes match packer expectations
                        scales_f32 = scales.to(torch.float32).contiguous()
                        M_u32 = M.to(torch.uint32).contiguous()

                        # packed: uint64 [G2, R, 2]
                        G32 = (qw_u8.shape[1] + 32 - 1) // 32
                        Wpair_u64 = forge.backend.cuda.kernels.pack_sparsegptq14_to_u64x2(
                            qw_u8, M_u32, scales_f32.view(qw_u8.shape[0], G32).transpose(0, 1).contiguous()
                        )

                        # ---- SAVE PACKED IN YOUR DESIRED LAYOUT ----
                        if args.save_dir:
                            # Detect proj type from handle_name
                            is_gate = handle_name.endswith("gate_proj")
                            is_up   = handle_name.endswith("up_proj")
                            is_down = handle_name.endswith("down_proj")

                            if is_gate:
                                eid = _parse_expert_id(handle_name)
                                _pending_gate_Wpair[(block_id, eid)] = Wpair_u64.detach().contiguous()

                            elif is_up:
                                eid = _parse_expert_id(handle_name)
                                key = (block_id, eid)
                                if key not in _pending_gate_Wpair:
                                    raise RuntimeError(f"Missing cached gate Wpair for block={block_id} expert={eid}. "
                                                    f"Loop order unexpected; need to pack+cache gate first.")

                                Wgate = _pending_gate_Wpair.pop(key)  # [G2, 2048, 2]
                                Wup   = Wpair_u64.detach().contiguous()  # [G2, 2048, 2]

                                # Stack along R -> [G2, 4096, 2]
                                W13 = torch.cat([Wgate, Wup], dim=1).contiguous()

                                # Save under gate_up_proj folder (Option B)
                                gateup_name = handle_name.replace("up_proj", "gate_up_proj")
                                out_dir = os.path.join(args.save_dir, f"block.{block_id}", gateup_name)
                                _save_Wpair_u64(os.path.join(out_dir, "Wpair_u64.pt"), W13)

                                del Wgate, Wup, W13

                            elif is_down:
                                out_dir = os.path.join(args.save_dir, f"block.{block_id}", handle_name)
                                _save_Wpair_u64(os.path.join(out_dir, "Wpair_u64.pt"), Wpair_u64.detach())

                            else:
                                # If you have non-expert or different names, decide what to do here.
                                pass

                        # aggressively free
                        del deq, qw_u8, scales_f32, M_u32, Wpair_u64
                        del qweight, scales, qzeros, M
                    else:
                        qweight, scales, qzeros = handle.quantize()

                        deq = forge.utils.engine.dequantize_forge_full(
                            handle.layer.weight.dtype,
                            qweight, scales, qzeros,
                            int(args.group_size), int(args.bits),
                        )
                        handle.layer.weight.copy_(deq)

                        if args.save_dir:
                            out_dir = os.path.join(args.save_dir, f"block.{block_id}", handle_name)
                            os.makedirs(out_dir, exist_ok=True)
                            torch.save(
                                {
                                    "qweight": qweight.to(torch.int8),        # pre-nibble
                                    "scales": scales.to(torch.float32),
                                    "qzeros": qzeros.to(torch.int32),
                                    "group_size": int(args.group_size),
                                    "bits": int(args.bits),
                                    "shape": tuple(handle.layer.weight.shape),
                                },
                                os.path.join(out_dir, "quantized_weight.pt"),
                            )
                        del deq, scales, qzeros

            for _, handle in handles.items():
                handle.reset()

            del handles
            del hooks
            torch.cuda.empty_cache()
            gc.collect()

        #Activation Update
        with torch.inference_mode():
            X = forge.utils.engine.forward(block, X, position_ids, N, B, device, offload_device, True, rotary_emb)

        if args.offload:
            #X.to(offload_device)
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