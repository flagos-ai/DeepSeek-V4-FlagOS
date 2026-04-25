# DeepSeek Inference Code Based on FlagOS

## Features

### FlagGems Acceleration
Enable [FlagGems](https://github.com/FlagOpen/FlagGems) operator acceleration by setting the environment variable `USE_FLAGGEMS=1`.

### O-Groups Grouped Projection Communication
When the model parallel size (MP) is greater than `o_groups`, set the environment variable `USE_OGROUPS_COMM=1` to enable grouped projection communication optimization for `wo_a` / `wo_b` (pair_comm_group and projection_comm_group). If MP <= o_groups, enabling this option will raise an error.

### FP8/FP4 → BF16 Weight Conversion Tool
Supports dequantizing DeepSeek-V3.2 quantized weights (MXFP4 E2M1 / FP8 E4M3) directly to BF16 format, implemented in pure PyTorch without depending on `kernel.py`.

### INT8 MoE Expert Quantization
Quantize MoE expert weights from BF16 to INT8 (per-channel symmetric) using `quantize_int8_moe.py`, then run inference with `config_pro_v4_int8.json`.

---

## Installation

```bash
# Install base dependencies
pip install -r requirements.txt

# Install FlagGems
pip install flag-gems==5.0.2

# Install FlagTree (NVIDIA platform example; for other chips see https://github.com/flagos-ai/flagtree):
python3 -m pip uninstall -y triton
python3 -m pip install flagtree===0.5.0 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple

```

---

## Weight Conversion

### Option 1: Convert from HuggingFace Format (Standard Flow)

```bash
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

When MP > o_groups and grouped projection communication is needed:

```bash
export USE_OGROUPS_COMM=1
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8
```

To use FP8 expert weights, remove `"expert_dtype": "fp4"` from `config_flash_v4.json` and specify `--expert-dtype fp8` in `convert.py`.

### Option 2: FP8/FP4 Quantized Weights → BF16

Follow the convert_weight.sh script:

```bash
# Step1: fp4/fp8 -> bf16
python3 convert_weight.py \
    --input-fp4-hf-path path-to-fp4-or-fp8-ckpt \
    --output-bf16-hf-path path-to-bf16-ckpt

# Step2: bf16 -> bf16-mp16
export MP=16
export HF_CKPT_PATH=path-to-bf16-ckpt
export SAVE_PATH=path-to-bf16-mp16-ckpt

export EXPERTS=256
export USE_OGROUPS_COMM=1
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8
```

### Option 3: INT8 MoE Expert Quantization (BF16 → INT8)

Quantize MoE expert weights to INT8, then shard for model-parallel inference.

```bash
# Step 1: Quantize expert weights BF16 -> INT8
python3 quantize_int8_moe.py \
    --input_dir path-to-bf16-hf-ckpt \
    --output_dir path-to-int8-hf-ckpt \
    --config_path config_pro_v4_int8.json

# Step 2: Shard for model-parallel
export MP=16
python convert.py \
    --hf-ckpt-path path-to-int8-hf-ckpt \
    --save-path path-to-int8-mp16-ckpt \
    --n-experts 384 \
    --model-parallel ${MP} \
    --expert-dtype int8 \
    --o-groups 16
```

Use `config_pro_v4_int8.json` for inference, which includes `quantization_config` to enable INT8 dequantization at runtime.

---

## Inference

### Interactive Chat

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive --temperature ${T}
```

### Batch Inference from File

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

### Single Node 8-GPU (MP8, with FlagGems)

```bash
bash run_mp8.sh
```

Equivalent command:

```bash
export USE_FLAGGEMS=1
torchrun --nproc-per-node 8 generate.py \
    --max-new-tokens 28 \
    --config config_flash_v4.json \
    --input-file prompt.txt \
    --ckpt-path path-to-bf16-mp8-ckpt
```

Note: MP=8 equals the default o_groups=8, which does not meet the USE_OGROUPS_COMM requirement, so it should not be set.

### Two-Node 16-GPU (MP16, with FlagGems + O-Groups Communication)

On node 0:

```bash
bash run_node_0.sh
```

On node 1:

```bash
bash run_node_1.sh
```

Replace `--master_addr` and `--master_port` in the scripts with actual values before running. MP=16 > o_groups=8, so `USE_OGROUPS_COMM=1` is already set in the scripts.

### General Multi-Node Inference

```bash
# When MP > o_groups, add: export USE_OGROUPS_COMM=1
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR \
    generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

### INT8 Quantized Model Inference

```bash
# Use config_pro_v4_int8.json which contains quantization_config
torchrun --nproc-per-node ${MP} generate.py \
    --ckpt-path path-to-int8-mp16-ckpt \
    --config config_pro_v4_int8.json \
    --input-file prompt.txt
```

---

## Config Reference

| Config | Model | expert_dtype | quantization_config | Description |
|--------|-------|-------------|---------------------|-------------|
| `config_flash_v4.json` | V4-Flash | fp4 | — | Flash model, 256 experts, o_groups=8 |
| `config_pro_v4.json` | V4-Pro | fp4 | — | Pro model, 384 experts, o_groups=16 |
| `config_pro_v4_int8.json` | V4-Pro | fp4 | linear_int8 | Pro model with INT8 quantized MoE experts |

- `expert_dtype`: controls how expert weights are stored on disk (fp4/fp8/int8)
- `quantization_config`: when present, enables INT8 dequantization for MoE experts at runtime
