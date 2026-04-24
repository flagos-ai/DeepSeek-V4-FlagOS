# 基于FlagOS的DeepSeek推理代码

## 新增功能

### FlagGems 加速支持
通过设置环境变量 `USE_FLAGGEMS=1` 启用 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子加速。

### FP8/FP4 → BF16 权重转换工具
支持将 DeepSeek-V3.2 的量化权重（MXFP4 E2M1 / FP8 E4M3）直接反量化为 BF16 格式，无需依赖 `kernel.py`，纯 PyTorch 实现。

### 模型并行分片优化
`convert.py` 针对 `wo_a` / `wo_b` 权重新增分组投影分片逻辑，支持更大规模的模型并行。

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

如需使用 FP8 专家权重，去掉 `config_flash_v4.json` 中的 `"expert_dtype": "fp4"` 并在 `convert.py` 中指定 `--expert-dtype fp8`。

### 方式二：FP8/FP4 量化权重转 BF16（新增）

按参考convert_weight.sh脚本流程执行：

```bash
# Step1: fp4/fp8 -> bf16
python3 convert_weight.py \
    --input-fp8-hf-path path-to-fp4-or-fp8-ckpt \
    --output-bf16-hf-path path-to-bf16-ckpt

# Step2: bf16 -> bf16-mp16
export MP=16
export HF_CKPT_PATH=path-to-bf16-ckpt
export SAVE_PATH=path-to-bf16-mp16-ckpt

export EXPERTS=256
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

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

### 双节点 16-GPU（MP16，启用 FlagGems）

在 node 0 上运行：

```bash
bash run_node_0.sh
```

在 node 1 上运行：

```bash
bash run_node_1.sh
```

运行前需在脚本中将 `--master_addr` 和 `--master_port` 替换为实际地址。

### 通用多节点推理

```bash
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR \
    generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

---

## 多芯片支持

本项目通过环境变量 `VENDOR_PATCH` 支持不同硬件平台。启动推理时设置对应的值即可，代码会自动切换到该平台的适配路径。

### 华为 Ascend A3

需要安装 `torch_npu` 及 Ascend 驱动。启动时设置 `VENDOR_PATCH=ascend`：

```bash
VENDOR_PATCH=ascend torchrun --nproc-per-node ${MP} generate.py \
    --ckpt-path ${SAVE_PATH} --config config_flash_v4.json --input-file ${FILE} --max-new-tokens 30
```

权重需提前转换为 BF16 MP16 格式（参见上方参数转换章节）。
