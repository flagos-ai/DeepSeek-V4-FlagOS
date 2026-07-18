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

# two-node 16-GPU generation script for MP16 model. Run this script on node 1.
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

export USE_FLAGGEMS=1
export USE_OGROUPS_COMM=1


torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=xxxx \
  --master_port=xxxx \
    generate.py --ckpt-path path-to-bf16-mp16-ckpt --config config_flash_v4.json --input-file prompt.txt --max-new-tokens 48
