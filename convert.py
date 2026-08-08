# Modified from https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/tree/master/inference/convert.py
import os
import shutil
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
    """
    Casts a tensor from e2m1fn to e4m3fn losslessly.
    """
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low  = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)
    # 6.0 * 2^6 = 384 < 448; 6.0 * 2^7 = 768 > 448; so MAX_OFFSET_BITS = 6
    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    # bOut, bIn, 128, 128
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    # bOut, bIn, 128*4
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    ## bOut, bIn, 1
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    # bOut, bIn, 128*4
    offset = scale / scale_max_offset_bits
    # bOut, bIn, 128, 128
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


mapping = {
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


def process_shard(rank, mp, hf_ckpt_path, save_path, n_experts, expert_dtype, file_paths, threads_per_proc, o_groups, use_ogroups_comm):
    torch.set_num_threads(threads_per_proc)
    n_local_experts = n_experts // mp
    state_dict = {}

    for file_path in file_paths:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
                    continue
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                if expert_dtype == "int8":
                    name = name.replace(".weight.scale", ".scale")  # int8 quantized scale
                name = name.replace("e_score_correction_bias", "bias")
                if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
                    key = name.split(".")[-1]
                else:
                    key = name.split(".")[-2]
                if key in mapping:
                    new_key, dim = mapping[key]
                else:
                    new_key, dim = key, None
                name = name.replace(key, new_key)

                new_param = param
                if "experts" in name and "shared_experts" not in name:
                    idx = int(name.split(".")[-3])
                    if idx < rank * n_local_experts or idx >= (rank + 1) * n_local_experts:
                        continue
                elif dim is not None:
                    if use_ogroups_comm and ("wo_a" in name or "wo_b" in name):
                        num_projection_groups = mp // o_groups
                        new_mp = mp // num_projection_groups
                        new_i = rank // num_projection_groups
                        shard_size = param.size(dim) // new_mp
                        new_param = param.narrow(dim, new_i * shard_size, shard_size).contiguous()
                    else:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, rank * shard_size, shard_size).contiguous()
                state_dict[name] = new_param

    save_file(state_dict, os.path.join(save_path, f"model{rank}-mp{mp}.safetensors"))
    print(f"Shard {rank}/{mp} done.", flush=True)


def _process_shard_wrapper(args):
    return process_shard(*args)


def main(hf_ckpt_path, save_path, n_experts, mp, expert_dtype, o_groups=8, num_workers=None):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        o_groups (int): Number of output projection groups.
        num_workers (int): Max parallel processes. Each worker holds its own shard in memory,
                           so peak memory ~ num_workers x shard_size. Defaults to mp.

    Returns:
        None
    """
    use_ogroups_comm = os.getenv("USE_OGROUPS_COMM", "0").lower() in ("1", "true", "yes")
    if use_ogroups_comm:
        if mp <= o_groups:
            raise ValueError(
                f"USE_OGROUPS_COMM requires model-parallel ({mp}) > o_groups ({o_groups}). "
                f"Please increase --model-parallel or unset USE_OGROUPS_COMM."
            )

    if num_workers is None:
        num_workers = mp

    file_paths = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    os.makedirs(save_path, exist_ok=True)

    total_threads = os.cpu_count() or 8
    threads_per_proc = max(1, total_threads // num_workers)

    args_list = [
        (i, mp, hf_ckpt_path, save_path, n_experts, expert_dtype, file_paths, threads_per_proc, o_groups, use_ogroups_comm)
        for i in range(mp)
    ]
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_process_shard_wrapper, args_list), total=mp, desc="Converting shards"):
            pass

    for file in ["tokenizer.json", "tokenizer_config.json"]:
        old_file_path = os.path.join(hf_ckpt_path, file)
        new_file_path = os.path.join(save_path, file)
        if os.path.exists(old_file_path):
            shutil.copyfile(old_file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    parser.add_argument("--expert-dtype", type=str, choices=["fp8", "fp4", "int8"], required=False, default=None)
    parser.add_argument("--o-groups", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Max parallel processes (default: same as --model-parallel). "
                             "Each worker holds its own shard in memory, so peak memory ~ num_workers x shard_size. "
                             "Reduce if OOM.")
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel, args.expert_dtype, args.o_groups, args.num_workers)
