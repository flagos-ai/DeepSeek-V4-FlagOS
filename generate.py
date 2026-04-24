# Modified from https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/tree/master/inference/generate.py
import os
import json
from argparse import ArgumentParser
from typing import List

if os.environ.get('VENDOR_PATCH') == 'ascend':
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """Batch generation with left-padded prompts.

    The first forward pass processes [min_prompt_len:] tokens (prefill phase).
    Subsequent passes generate one token at a time (decode phase). For positions
    still within a prompt, the ground-truth token overrides the model's prediction.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if os.environ.get('VENDOR_PATCH') == 'ascend':
        import flag_gems
        flag_gems.only_enable(record=True, once=True, path="/root/gems.txt", include=["ne_scalar", "lt_scalar", "ge_scalar"])
    elif os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
        import flag_gems
        flag_gems.enable(record=True, once=True, path="/tmp/gems.txt")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    if os.environ.get('VENDOR_PATCH') != 'ascend':
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    pair_comm_group = None
    projection_comm_group = None
    if world_size > 1:
        print(f"Initializing distributed process group: world_size={world_size}, rank={rank}, local_rank={local_rank}", flush=True)
        init_process_group_kwargs = {
            'backend': "nccl",
            'world_size': world_size,
            'rank': rank,
        }
        dist.init_process_group(**init_process_group_kwargs)
        cur_rank = dist.get_rank()

        pair_group_size = world_size // 8

        pair_groups = []
        for i in range(0, world_size, pair_group_size):
            ranks = list(range(i, min(i + pair_group_size, world_size)))
            pair_groups.append(dist.new_group(ranks=ranks))

        pair_group_id = cur_rank // pair_group_size
        pair_comm_group = pair_groups[pair_group_id]
        pair_ranks = list(range(pair_group_id * pair_group_size, min((pair_group_id + 1) * pair_group_size, world_size)))
        print(f"cur_rank: {cur_rank}, pair_comm_group ranks: {pair_ranks}", flush=True)

        projection_group_count = world_size // 8

        proj_groups = []
        for i in range(projection_group_count):
            ranks = list(range(i, world_size, projection_group_count))
            proj_groups.append(dist.new_group(ranks=ranks))

        projection_group_id = cur_rank % projection_group_count
        projection_comm_group = proj_groups[projection_group_id]
        proj_ranks = list(range(projection_group_id, world_size, projection_group_count))
        print(f"cur_rank: {cur_rank}, projection_comm_group ranks: {proj_ranks}", flush=True)

        dummy = torch.zeros(1, device="cuda")

        dist.all_reduce(dummy, group=pair_comm_group)
        torch.cuda.synchronize()
        print(f"cur_rank: {cur_rank}: Initialized pair_comm_group NCCL communicator", flush=True)

        # Then initialize projection_comm_group on ALL ranks
        dist.all_reduce(dummy, group=projection_comm_group)
        torch.cuda.synchronize()
        print(f"cur_rank: {cur_rank}: Initialized projection_comm_group NCCL communicator", flush=True)

        dist.barrier()
        print(f"cur_rank: {cur_rank}: All communicators initialized", flush=True)

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    if interactive:
        args.max_batch_size = 1
    print(args)
    with torch.device("cuda"):
        model = Transformer(args, pair_comm_group=pair_comm_group if world_size > 1 else None, projection_comm_group=projection_comm_group if world_size > 1 else None)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    if world_size > 1:
        dist.barrier()
    print(f"Rank {rank}: Model loaded, entering generation loop", flush=True)
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
