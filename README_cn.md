# 基于FlagOS的DeepSeek推理代码

## 新增功能

### FlagGems 加速支持
通过设置环境变量 `USE_FLAGGEMS=1` 启用 [FlagGems](https://github.com/FlagOpen/FlagGems) 算子加速。

### O-Groups 分组投影通信
当模型并行数（MP）大于 `o_groups` 时，可通过设置环境变量 `USE_OGROUPS_COMM=1` 启用 `wo_a` / `wo_b` 的分组投影通信优化（pair_comm_group 和 projection_comm_group）。若 MP <= o_groups，启用该选项会报错提示。

### FP8/FP4 → BF16 权重转换工具
支持将 DeepSeek-V3.2 的量化权重（MXFP4 E2M1 / FP8 E4M3）直接反量化为 BF16 格式，无需依赖 `kernel.py`，纯 PyTorch 实现。

### 数据格式选择
通过 `--data-format` 参数选择推理数据格式，支持 `bf16`（默认）和 `fp8`。`bf16` 模式下所有层使用 BF16 权重，QAT 量化模拟关闭；`fp8` 模式下使用原始 FP8 权重并启用 `act_quant`/`fp4_act_quant`（需要 `kernel` 模块）。

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
export USE_OGROUPS_COMM=1
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8
```

---

## 推理

### 交互式对话

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive --temperature ${T} --data-format bf16
```

### 文件批量推理

```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE} --data-format bf16
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
