# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Step1: fp4/fp8 -> bf16
python3 convert_weight.py \
	--input-fp4-hf-path path-to-fp4-or-fp8-ckpt \
    --output-bf16-hf-path path-to-bf16-ckpt

# Step2: bf16 -> bf16-mp16
export MP=16
export HF_CKPT_PATH=path-to-bf16-ckpt
export SAVE_PATH=path-to-bf16-mp16-ckpt

export EXPERTS=256
# 当 MP > o_groups 时，设置 USE_OGROUPS_COMM=1 启用分组投影通信
export USE_OGROUPS_COMM=1
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP} --o-groups 8
