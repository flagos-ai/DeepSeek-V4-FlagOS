# 基于FlagOS的DeepSeek推理代码

## 新增功能

### FlagGems 加速支持
通过设置环境变量 `USE_FLAGGEMS=1` 启用 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子加速。

### O-Groups 分组投影通信
当模型并行数（MP）大于 `o_groups` 时，可通过设置环境变量 `USE_OGROUPS_COMM=1` 启用 `wo_a` / `wo_b` 的分组投影通信优化（pair_comm_group 和 projection_comm_group）。若 MP <= o_groups，启用该选项会报错提示。

### FP8/FP4 → BF16 权重转换工具
支持将 DeepSeek-V3.2 的量化权重（MXFP4 E2M1 / FP8 E4M3）直接反量化为 BF16 格式，无需依赖 `kernel.py`，纯 PyTorch 实现。

### INT8 MoE 专家量化
支持将 BF16 模型的 MoE 专家权重量化为 INT8（逐通道对称量化），使用 `quantize_int8_moe.py` 量化后配合 `config_pro_v4_int8.json` 进行推理。

### 流式权重转换（内存优化）
新增 `convert_streaming.py`，针对超大模型（如 2T 参数）在有限内存下的转换场景进行优化。与 `convert.py` 功能一致，额外支持：
- 多进程并行转换（`--num-workers`）
- 可选 `--streaming` 模式，通过临时文件 + 增量保存避免将完整 shard 加载到内存
- 支持 `--o-groups` 分组投影分片和 `--expert-dtype int8`

---

## 安装依赖

```bash
# 安装原始依赖
pip install -r requirements.txt

# 安装 FlagGems
pip install flag-gems==5.0.2

# 安装FlagTree, 以英伟达平台为例, 其他芯片请参考https://github.com/flagos-ai/flagtree：
python3 -m pip uninstall -y triton
python3 -m pip install flagtree===0.5.0 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple

```

---

## 参数转换

### 方式一：从 HuggingFace 格式转换（原始流程）

```bash
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

当 MP > o_groups 且需要启用分组投影通信时：

```bash
export USE_OGROUPS_COMM=1
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8
```

如果内存不足（例如转换超大模型），可使用流式版本：

```bash
# 多进程并行转换
python convert_streaming.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} \
    --n-experts ${EXPERTS} --model-parallel ${MP} --num-workers 4

# 低内存流式模式（通过临时文件减少内存占用）
python convert_streaming.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} \
    --n-experts ${EXPERTS} --model-parallel ${MP} --num-workers 4 --streaming

# 支持 o-groups 分组投影分片（MP > o_groups 时）
python convert_streaming.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} \
    --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8 --num-workers 4
```

如需使用 FP8 专家权重，去掉 `config_flash_v4.json` 中的 `"expert_dtype": "fp4"` 并在 `convert.py` 中指定 `--expert-dtype fp8`。

### 方式二：FP8/FP4 量化权重转 BF16

参考 convert_weight.sh 脚本流程执行：

```bash
# 第一步：fp4/fp8 -> bf16
python3 convert_weight.py \
    --input-fp4-hf-path path-to-fp4-or-fp8-ckpt \
    --output-bf16-hf-path path-to-bf16-ckpt

# 第二步：bf16 -> bf16-mp16
export MP=16
export HF_CKPT_PATH=path-to-bf16-ckpt
export SAVE_PATH=path-to-bf16-mp16-ckpt

export EXPERTS=256
export USE_OGROUPS_COMM=1
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8
```

### 方式三：INT8 MoE 专家量化（BF16 → INT8）

将 BF16 模型的 MoE 专家权重量化为 INT8，再分片用于模型并行推理。

```bash
# 第一步：将专家权重从 BF16 量化为 INT8
python3 quantize_int8_moe.py \
    --input_dir path-to-bf16-hf-ckpt \
    --output_dir path-to-int8-hf-ckpt \
    --config_path config_pro_v4_int8.json

# 第二步：分片用于模型并行
export MP=16
python convert.py \
    --hf-ckpt-path path-to-int8-hf-ckpt \
    --save-path path-to-int8-mp16-ckpt \
    --n-experts 384 \
    --model-parallel ${MP} \
    --expert-dtype int8 \
    --o-groups 16
```

推理时使用 `config_pro_v4_int8.json`，其中包含 `quantization_config` 字段以在运行时启用 INT8 反量化。

---

## 推理

### 交互式对话

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive --temperature ${T}
```

### 文件批量推理

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

### 单节点 8-GPU（MP8，启用 FlagGems）

```bash
bash run_mp8.sh
```

等价命令：

```bash
export USE_FLAGGEMS=1
torchrun --nproc-per-node 8 generate.py \
    --max-new-tokens 28 \
    --config config_flash_v4.json \
    --input-file prompt.txt \
    --ckpt-path path-to-bf16-mp8-ckpt
```

注意：MP=8 等于默认 o_groups=8，不满足 USE_OGROUPS_COMM 的启用条件，无需设置。

### 双节点 16-GPU（MP16，启用 FlagGems + O-Groups 通信）

在 node 0 上运行：

```bash
bash run_node_0.sh
```

在 node 1 上运行：

```bash
bash run_node_1.sh
```

运行前需在脚本中将 `--master_addr` 和 `--master_port` 替换为实际地址。MP=16 > o_groups=8，脚本中已设置 `USE_OGROUPS_COMM=1`。

### 通用多节点推理

```bash
# 当 MP > o_groups 时，添加 export USE_OGROUPS_COMM=1
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR \
    generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

### INT8 量化模型推理

```bash
# 使用包含 quantization_config 的 config_pro_v4_int8.json
torchrun --nproc-per-node ${MP} generate.py \
    --ckpt-path path-to-int8-mp16-ckpt \
    --config config_pro_v4_int8.json \
    --input-file prompt.txt
```

---

## 配置文件说明

| 配置文件 | 模型 | expert_dtype | quantization_config | 说明 |
|---------|------|-------------|---------------------|------|
| `config_flash_v4.json` | V4-Flash | fp4 | — | Flash 模型，256 专家，o_groups=8 |
| `config_pro_v4.json` | V4-Pro | fp4 | — | Pro 模型，384 专家，o_groups=16 |
| `config_pro_v4_int8.json` | V4-Pro | fp4 | linear_int8 | Pro 模型，MoE 专家权重 INT8 量化 |

- `expert_dtype`：控制专家权重在磁盘上的存储格式（fp4/fp8/int8）
- `quantization_config`：存在时，在运行时启用 MoE 专家的 INT8 反量化
