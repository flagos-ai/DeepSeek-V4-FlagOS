"""
DeepSeek-MOE Expert Weight Quantization Script
Linearly quantize MOE layer expert weights from bf16 to int8 (per-channel symmetric)

Quantization method:
  scale = max(|W|, dim=-1) / 127   (per output-channel)
  W_int8 = clamp(round(W / scale), -128, 127)

Output:
  - Quantized safetensors shards (expert weights in int8, others remain bf16)
  - A scale tensor for each quantized weight, named {original_name}.scale
  - Updated index.json and config.json
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

# Regex pattern matching MOE expert weights
MOE_EXPERT_PATTERN = re.compile(
    r"layers\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)\.weight"
)


def quantize_int8_per_channel(weight: torch.Tensor):
    """
    Per-channel (output dim) symmetric linear quantization bf16 -> int8

    Args:
        weight: shape (out_features, in_features), dtype=bf16
    Returns:
        w_int8: shape (out_features, in_features), dtype=int8
        scale:  shape (out_features,), dtype=bf16
    """
    # Convert to float32 for computation, avoiding bf16 precision issues
    w = weight.float()
    # per-channel absmax
    absmax = w.abs().amax(dim=-1)  # (out_features,)
    # Avoid division by zero
    absmax = absmax.clamp(min=1e-10)
    scale = absmax / 127.0
    # Quantize
    w_int8 = (w / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    scale = scale.to(torch.bfloat16)
    return w_int8, scale


def main():
    parser = argparse.ArgumentParser(description="Quantize DeepSeek MOE experts to int8")
    parser.add_argument("--input_dir", type=str, required=True, help="Original bf16 model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Quantized model output directory")
    parser.add_argument("--config_path", type=str, required=True, help="Model inference config")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config_path



    # Read index
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Group by shard
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
        # Free memory
        del tensors, new_tensors

    # Write index
    new_index = {
        "metadata": index["metadata"],
        "weight_map": new_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy config and add quantization info

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

    # Copy other necessary files
    for fname in os.listdir(input_dir):
        if fname.endswith((".json", ".py", ".model", ".tiktoken", "jinja")) and fname not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            src = input_dir / fname
            dst = output_dir / fname
            if src.is_file() and not dst.exists():
                shutil.copy2(str(src), str(dst))

    print("Done! Quantized model saved to:", output_dir)


if __name__ == "__main__":
    main()
