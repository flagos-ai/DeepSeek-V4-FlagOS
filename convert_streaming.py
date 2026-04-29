"""
Streaming model weight conversion script with multiprocessing support.

Features:
1. Multi-process parallel conversion for speed
2. Optional --streaming mode for limited-memory machines (uses temp files to avoid holding full shard in memory)

Usage is identical to convert.py:
    python convert_streaming.py --hf-ckpt-path <path> --save-path <path> \
        --n-experts 256 --model-parallel 8 --num-workers 4

For low-memory machines, add --streaming:
    python convert_streaming.py --hf-ckpt-path <path> --save-path <path> \
        --n-experts 256 --model-parallel 8 --num-workers 4 --streaming
"""

import os
import shutil
import tempfile
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

import torch
from safetensors.torch import safe_open, save_file


FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Casts a tensor from e2m1fn to e4m3fn losslessly."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def process_tensor_name(name: str, rank: int, mp: int, n_local_experts: int, expert_dtype: str = None):
    """
    Process tensor name and determine whether this rank needs it.

    Returns:
        (new_name, slice_info) or None if this rank doesn't need this tensor.
        slice_info: (dim, rank, mp) or None (no sharding needed)
    """
    if name.startswith("model."):
        name = name[len("model."):]

    if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
        return None

    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    if expert_dtype == "int8":
        name = name.replace(".weight.scale", ".scale")
    name = name.replace("e_score_correction_bias", "bias")

    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]

    if key in MAPPING:
        new_key, dim = MAPPING[key]
    else:
        new_key, dim = key, None

    name = name.replace(key, new_key)

    if "experts" in name and "shared_experts" not in name:
        parts = name.split(".")
        experts_pos = parts.index("experts")
        idx = int(parts[experts_pos + 1])
        if idx < rank * n_local_experts or idx >= (rank + 1) * n_local_experts:
            return None
        return name, None

    if dim is not None:
        return name, (dim, rank, mp)

    return name, None


def get_sharded_tensor(param: torch.Tensor, slice_info: tuple, name: str,
                       use_ogroups_comm: bool, o_groups: int) -> torch.Tensor:
    """Shard tensor according to slice_info."""
    dim, i, mp = slice_info
    if use_ogroups_comm and ("wo_a" in name or "wo_b" in name):
        num_projection_groups = mp // o_groups
        new_mp = mp // num_projection_groups
        new_i = i // num_projection_groups
        shard_size = param.size(dim) // new_mp
        return param.narrow(dim, new_i * shard_size, shard_size).contiguous()
    else:
        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
        shard_size = param.size(dim) // mp
        return param.narrow(dim, i * shard_size, shard_size).contiguous()


def process_single_file(file_path: str, rank: int, mp: int, n_local_experts: int,
                        expert_dtype: str, use_ogroups_comm: bool, o_groups: int) -> dict:
    """Process a single safetensors file and return tensors needed by this rank."""
    result = {}

    with safe_open(file_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            process_result = process_tensor_name(name, rank, mp, n_local_experts, expert_dtype)
            if process_result is None:
                continue

            new_name, slice_info = process_result
            param = f.get_tensor(name)

            if slice_info is not None:
                param = get_sharded_tensor(param, slice_info, new_name, use_ogroups_comm, o_groups)

            result[new_name] = param

    return result


def _process_rank(args_tuple):
    """Worker: process one rank, in-memory mode (fast)."""
    rank, mp, hf_ckpt_path, save_path, n_experts, expert_dtype, file_paths, threads_per_proc, o_groups, use_ogroups_comm = args_tuple
    torch.set_num_threads(threads_per_proc)
    n_local_experts = n_experts // mp
    state_dict = {}

    for file_path in file_paths:
        tensors = process_single_file(
            file_path, rank, mp, n_local_experts,
            expert_dtype, use_ogroups_comm, o_groups
        )
        state_dict.update(tensors)
        del tensors

    output_file = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
    save_file(state_dict, output_file)
    return rank


def _process_rank_streaming(args_tuple):
    """Worker: process one rank, streaming mode (low memory)."""
    rank, mp, hf_ckpt_path, save_path, n_experts, expert_dtype, file_paths, threads_per_proc, o_groups, use_ogroups_comm = args_tuple
    torch.set_num_threads(threads_per_proc)
    n_local_experts = n_experts // mp

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = []

        for batch_idx, file_path in enumerate(file_paths):
            tensors = process_single_file(
                file_path, rank, mp, n_local_experts,
                expert_dtype, use_ogroups_comm, o_groups
            )
            if tensors:
                temp_file = os.path.join(temp_dir, f"batch_{batch_idx}.safetensors")
                save_file(tensors, temp_file)
                temp_files.append(temp_file)
                del tensors

        merged = {}
        for temp_file in temp_files:
            with safe_open(temp_file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    merged[name] = f.get_tensor(name)

        output_file = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
        save_file(merged, output_file)

    return rank


def main(hf_ckpt_path: str, save_path: str, n_experts: int, mp: int,
         expert_dtype: str = None, o_groups: int = 8, num_workers: int = 4,
         streaming: bool = False):
    torch.set_num_threads(8)

    use_ogroups_comm = os.getenv("USE_OGROUPS_COMM", "0").lower() in ("1", "true", "yes")
    if use_ogroups_comm:
        if mp <= o_groups:
            raise ValueError(
                f"USE_OGROUPS_COMM requires model-parallel ({mp}) > o_groups ({o_groups}). "
                f"Please increase --model-parallel or unset USE_OGROUPS_COMM."
            )

    file_paths = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    os.makedirs(save_path, exist_ok=True)

    total_threads = os.cpu_count() or 8
    threads_per_proc = max(1, total_threads // num_workers)

    args_list = [
        (i, mp, hf_ckpt_path, save_path, n_experts, expert_dtype, file_paths, threads_per_proc, o_groups, use_ogroups_comm)
        for i in range(mp)
    ]

    worker_fn = _process_rank_streaming if streaming else _process_rank
    mode_str = "streaming" if streaming else "in-memory"
    print(f"Converting with {num_workers} workers ({mode_str} mode)")

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(worker_fn, args_list), total=mp, desc="Converting shards"):
            pass

    for file in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(hf_ckpt_path, file)
        dst = os.path.join(save_path, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    print(f"Conversion complete! Output saved to: {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    parser.add_argument("--expert-dtype", type=str, choices=["fp8", "fp4", "int8"], required=False, default=None)
    parser.add_argument("--o-groups", type=int, default=8)
    parser.add_argument("--num-workers", type=int, required=True,
                        help="Number of parallel processes. Each worker holds one shard in memory.")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode with temp files to reduce memory usage.")
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel,
         args.expert_dtype, args.o_groups, args.num_workers, args.streaming)
