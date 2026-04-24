# Single node 8-GPU generation script for MP8 model.
export USE_FLAGGEMS=1

torchrun --nproc-per-node 8 \
         generate.py \
         --max-new-tokens 28 \
         --config config_flash_v4.json \
         --input-file prompt.txt \
         --ckpt-path path-to-bf16-mp8-ckpt \
         --data-format bf16