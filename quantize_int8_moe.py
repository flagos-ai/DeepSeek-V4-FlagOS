"""
DeepSeek-MOE Expert 权重量化脚本
将 MOE 层的 expert 权重从 bf16 线性量化到 int8 (per-channel symmetric)

量化方式：
  scale = max(|W|, dim=-1) / 127   (per output-channel)
  W_int8 = clamp(round(W / scale), -128, 127)

输出：
  - 量化后的 safetensors 分片（expert 权重为 int8，其余保持 bf16）
  - 每个量化权重对应一个 scale tensor，命名为 {原名}.scale
  - 更新后的 index.json 和 config.json
"""

import argparse
import json
import os
import re
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# 匹配 MOE expert 权重的正则
MOE_EXPERT_PATTERN = re.compile(
    r"layers\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)\.weight"
)


def quantize_int8_per_channel(weight: torch.Tensor):
    """
    Per-channel (output dim) 对称线性量化 bf16 -> int8

    Args:
        weight: shape (out_features, in_features), dtype=bf16
    Returns:
        w_int8: shape (out_features, in_features), dtype=int8
        scale:  shape (out_features,), dtype=bf16
    """
    # 转 float32 计算，避免 bf16 精度问题
    w = weight.float()
    # per-channel absmax
    absmax = w.abs().amax(dim=-1)  # (out_features,)
    # 避免除零
    absmax = absmax.clamp(min=1e-10)
    scale = absmax / 127.0
    # 量化
    w_int8 = (w / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    scale = scale.to(torch.bfloat16)
    return w_int8, scale


def main():
    parser = argparse.ArgumentParser(description="Quantize DeepSeek MOE experts to int8")
    parser.add_argument("--input_dir", type=str, required=True, help="原始 bf16 模型目录")
    parser.add_argument("--output_dir", type=str, required=True, help="量化后模型输出目录")
    parser.add_argument("--config_path", type=str, required=True, help="模型推理时的config")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config_path

    config_src = Path(args.config_path)
    config_dst = output_dir / "config.json"



    # 读取 index
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # 按 shard 分组
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    new_weight_map = {}
    shards = sorted(shard_to_keys.keys())

    for shard_name in tqdm(shards, desc="Processing shards"):
        shard_path = input_dir / shard_name
        tensors = load_file(str(shard_path), device="cpu")
        new_tensors = {}

        for key in shard_to_keys[shard_name]:
            tensor = tensors[key]
            if MOE_EXPERT_PATTERN.match(key):
                
                w_int8, scale = quantize_int8_per_channel(tensor)
                new_tensors[key] = w_int8
                new_tensors[key + ".scale"] = scale
                new_weight_map[key] = shard_name
                new_weight_map[key + ".scale"] = shard_name
            else:
                new_tensors[key] = tensor
                new_weight_map[key] = shard_name

        save_file(new_tensors, str(output_dir / shard_name))
        # 释放内存
        del tensors, new_tensors

    # 写 index
    new_index = {
        "metadata": index["metadata"],
        "weight_map": new_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # 复制 config 并添加量化信息

    with open(config_path) as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "linear_int8",
        "target": "moe_experts",
        "scheme": "per_channel_symmetric",
        "bits": 8,
        "scale_suffix": ".scale",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 复制其他必要文件
    for fname in os.listdir(input_dir):
        if fname.endswith((".json", ".py", ".model", ".tiktoken", "jinja")) and fname not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            src = input_dir / fname
            dst = output_dir / fname
            if src.is_file() and not dst.exists():
                import shutil
                shutil.copy2(str(src), str(dst))

    print("Done! 量化模型已保存到:", output_dir)


if __name__ == "__main__":
    main()
