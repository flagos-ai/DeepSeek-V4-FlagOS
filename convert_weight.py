# Based on https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py
"""
DeepSeek-V4 FP4/FP8 -> BF16 Converter
This script converts DeepSeek-V4 FP4/FP8 quantized checkpoints to BF16 format by dequantizing weights using the corresponding scales. It handles both non-expert (FP8 E4M3) and expert (MXFP4 E2M1) weights, as well as the special case of DSA indexer weights and MTP layers.
"""

import os
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file


BLOCK_SIZE = 128
FP4_GROUP_SIZE = 32

# E2M1 FP4 lookup table: 4-bit index -> float value
# Bit layout: sign(1) | exponent(2) | mantissa(1), bias=1
_FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.bfloat16)


def dequant_fp4_weight(weight_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 weight to bf16 by unpacking FP4 E2M1 nibbles via LUT.

    Each int8 byte stores 2 FP4 values (low nibble + high nibble).
    float4_e2m1fn_x2 does NOT support .to(bfloat16), so we manually unpack.
    Scale is E8M0, one per group of 32 elements along the last dimension.

    Args:
        weight_packed: [out_features, in_features/2], int8, each byte = 2 FP4 values
        scale_e8m0:    [out_features, in_features/32], float8_e8m0fnu, E8M0 scale per group of 32
    Returns:
        bf16 tensor [out_features, in_features]
    """
    out_features, packed_in = weight_packed.shape
    in_features = packed_in * 2

    # Unpack two FP4 values per byte via nibble extraction + LUT
    raw = weight_packed.to(torch.uint8)
    low_nibble = (raw & 0x0F).to(torch.long)
    high_nibble = (raw >> 4).to(torch.long)
    lut = _FP4_E2M1_LUT.to(weight_packed.device)
    low_vals = lut[low_nibble]
    high_vals = lut[high_nibble]
    # Interleave: [low_0, high_0, low_1, high_1, ...]
    fp4_values = torch.stack([low_vals, high_vals], dim=-1).reshape(out_features, in_features)

    # Decode E8M0 scale and expand to match fp4_values
    scale = decode_e8m0_scale(scale_e8m0)
    if scale.dim() == 2 and scale.shape[0] == out_features:
        # Scale already shaped [out_features, num_groups_per_row]
        num_groups_per_row = scale.shape[1]
    else:
        # Flat scale: compute groups per row from total count
        total_scales = scale.numel()
        num_groups_per_row = total_scales // out_features
        scale = scale.reshape(out_features, num_groups_per_row)
    actual_group_size = in_features // num_groups_per_row
    scale = scale.unsqueeze(-1).expand(-1, -1, actual_group_size).reshape(out_features, in_features)

    return (fp4_values * scale).to(torch.bfloat16)


def is_expert_weight(name: str) -> bool:
    """Check if a weight belongs to an expert (MoE) layer, excluding shared_experts."""
    return "experts" in name and "shared_experts" not in name


def decode_e8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Decode E8M0 (unsigned 8-bit exponent-only) scale to float32.
    E8M0 stores only the exponent: value = 2^(exp - 127), same as IEEE 754 exponent encoding.
    If scale is already float32, return as-is.
    """
    if scale.dtype == torch.float32:
        return scale
    if scale.dtype in (torch.bfloat16, torch.float16):
        return scale.float()
    # float8_e8m0fnu: .to(int32) does value conversion not bit reinterpret,
    # so use native .float() which handles E8M0 correctly
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.float()
    # uint8 / int8 raw E8M0 bytes: interpret as IEEE 754 exponent
    if scale.element_size() == 1:
        # Reconstruct float32 from exponent: set exponent bits in IEEE 754 float32
        # float32 = sign(1) + exponent(8) + mantissa(23)
        # E8M0 value stored as raw exponent byte -> float = 2^(byte - 127)
        exp_bits = scale.to(torch.int32) << 23
        return exp_bits.view(torch.float32)
    return scale.float()


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """
    Dequantize FP8 weight to BF16 using block-wise scale.
    Based on V3.2 model.py:490-495, works on both CPU and CUDA.

    Each (block_size x block_size) block of the weight is multiplied by one scale value.
    Handles non-aligned dimensions (e.g. kv_a_proj_with_mqa shape 576x7168, 576 % 128 != 0)
    by padding to the nearest multiple of block_size, dequantizing, then trimming.
    """
    shape = weight.shape
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    M, N = shape

    # Decode E8M0 scale if needed
    scale = decode_e8m0_scale(scale)

    # Pad to nearest multiple of block_size if needed
    pad_m = (block_size - M % block_size) % block_size
    pad_n = (block_size - N % block_size) % block_size
    if pad_m or pad_n:
        weight = torch.nn.functional.pad(weight, (0, pad_n, 0, pad_m))
    Mp, Np = weight.shape

    # V3.2 dequant: reshape into blocks, scale, reshape back
    weight = weight.view(
        Mp // block_size, block_size,
        Np // block_size, block_size
    ).transpose(1, 2).contiguous().view(-1, block_size * block_size)

    weight = (weight.float() * scale.reshape(-1, 1)).to(torch.bfloat16)

    weight = weight.view(
        Mp // block_size, Np // block_size,
        block_size, block_size
    ).transpose(1, 2).contiguous().view(Mp, Np)

    # Trim padding
    if pad_m or pad_n:
        weight = weight[:M, :N]

    return weight


def main(input_path, bf16_path, device="cuda"):
    torch.set_default_dtype(torch.bfloat16)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(bf16_path, exist_ok=True)

    # 1. Copy non-safetensor files (config.json, tokenizer, etc.)
    print("Copying auxiliary files...")
    for file_path in glob(os.path.join(input_path, "*")):
        fname = os.path.basename(file_path)
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        dst = os.path.join(bf16_path, fname)
        if os.path.isfile(file_path):
            shutil.copy2(file_path, dst)
            print(f"  Copied {fname}")

    # 2. Load model index
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # 3. Pre-build scale_inv lookup: weight_name -> scale_inv_name
    # V3.2 naming: "xxx.weight" -> "xxx.weight_scale_inv"
    scale_inv_map = {}
    all_scale_names = set()
    for name in weight_map:
        if name.endswith("scale"):
            weight_name = name[:-len("scale")] + "weight"
            if weight_name in weight_map:
                all_scale_names.add(name)
                scale_inv_map[weight_name] = name

    # Separate expert (FP4) vs non-expert (FP8) scale mappings
    fp4_scale_map = {k: v for k, v in scale_inv_map.items() if is_expert_weight(k)}
    fp8_scale_map = {k: v for k, v in scale_inv_map.items() if not is_expert_weight(k)}

    print(f"Model: DeepSeek (DeepseekForCausalLM)")
    print(f"Device: {device}")
    print(f"Total keys in index: {len(weight_map)}")
    print(f"FP4 expert weights with scale: {len(fp4_scale_map)}")
    print(f"FP8 non-expert weights with scale: {len(fp8_scale_map)}")
    print(f"Scale entries: {len(all_scale_names)}")

    # Cache for loaded safetensor files on CPU (handles cross-shard scale lookups)
    # All shards are loaded to CPU to avoid GPU OOM; individual tensors are moved to
    # GPU for dequant one at a time.
    loaded_files = {}
    use_cuda = (device == "cuda")

    def get_tensor(tensor_name):
        """Load tensor from the correct shard file (CPU), with caching."""
        file_name = weight_map.get(tensor_name)
        if file_name is None:
            raise KeyError(tensor_name)
        if file_name not in loaded_files:
            file_path = os.path.join(input_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    # 4. Process safetensor files one by one
    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    converted_count = 0
    kept_count = 0

    for safetensor_file in tqdm(safetensor_files, desc="Converting FP4/FP8 -> BF16"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            # Skip scale_inv tensors (will be removed from output)
            if weight_name in all_scale_names:
                continue

            if weight.element_size() == 1 and weight_name in scale_inv_map:
                scale_inv_name = scale_inv_map[weight_name]
                try:
                    scale_inv = get_tensor(scale_inv_name)
                    if is_expert_weight(weight_name):
                        # FP4 expert weight -> dequantize using MXFP4 logic
                        # print(f"  FP4 dequant: {weight_name}, weight={weight.shape}, scale={scale_inv.shape}, dtype={scale_inv.dtype}")
                        if use_cuda:
                            result = dequant_fp4_weight(weight.cuda(), scale_inv.cuda())
                            new_state_dict[weight_name] = result.cpu()
                        else:
                            new_state_dict[weight_name] = dequant_fp4_weight(weight, scale_inv)
                    else:
                        # FP8 non-expert weight -> dequantize using block-wise scale
                        if use_cuda:
                            result = weight_dequant(weight.cuda(), scale_inv.cuda())
                            new_state_dict[weight_name] = result.cpu()
                        else:
                            new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                    converted_count += 1
                except KeyError:
                    print(f"  Warning: scale '{scale_inv_name}' not loadable for {weight_name}, keeping raw")
                    new_state_dict[weight_name] = weight
            else:
                # BF16/FP32 weights: norms, biases, gate.weight, indexer.weights_proj,
                # indexer.k_norm, MTP layers (enorm, hnorm, eh_proj, shared_head), embed, lm_head, etc.
                new_state_dict[weight_name] = weight
                kept_count += 1

        # Save converted shard
        save_file(new_state_dict, os.path.join(bf16_path, file_name))

        # Memory management: keep at most 2 cached shard files on CPU
        # (needed for cross-shard scale lookups, e.g. weight in shard N, scale in shard N+1)
        while len(loaded_files) > 2:
            oldest = next(iter(loaded_files))
            del loaded_files[oldest]

    # 5. Update model index: remove all scale_inv entries
    new_weight_map = {k: v for k, v in weight_map.items() if k not in all_scale_names}
    new_index = {
        "metadata": model_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(bf16_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    print(f"\nDone!")
    print(f"  FP4/FP8 -> BF16 converted: {converted_count}")
    print(f"  Already BF16/FP32 (kept as-is): {kept_count}")
    print(f"  Scale entries removed: {len(all_scale_names)}")
    print(f"  Output keys: {len(new_weight_map)} (was {len(weight_map)})")
    print(f"  Output saved to: {bf16_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert DeepSeek-V4 FP4/FP8 checkpoint to BF16")
    parser.add_argument("--input-fp4-hf-path", type=str, required=True,
                        help="Path to the FP4/FP8 HuggingFace model directory (DeepSeek-V4)")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True,
                        help="Path to the output BF16 model directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for dequantization (default: cuda)")
    args = parser.parse_args()
    main(args.input_fp4_hf_path, args.output_bf16_hf_path, args.device)
