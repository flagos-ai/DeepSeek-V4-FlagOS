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

# Single node 8-GPU generation script for MP8 model.
export USE_FLAGGEMS=1

torchrun --nproc-per-node 8 \
         generate.py \
         --max-new-tokens 28 \
         --config config_flash_v4.json \
         --input-file prompt.txt \
         --ckpt-path path-to-bf16-mp8-ckpt
