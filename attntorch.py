import torch
import math


def sparse_attn_pytorch(
    q: torch.Tensor,       # (b, m, h, d) bf16
    kv: torch.Tensor,      # (b, n, d)    bf16
    attn_sink: torch.Tensor,  # (h,)      fp32
    topk_idxs: torch.Tensor,  # (b, m, topk) int32
    softmax_scale: float,
) -> torch.Tensor:
    b, s, h, d = q.size()

    # --- wrapper: pad heads to 16, matching tilelang wrapper ---
    orig_h = h
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
        h = 16

    topk = topk_idxs.shape[-1]
    block = 64
    num_blocks = math.ceil(topk / block)

    # q_shared: (b, m, h, d) bf16 — keep bf16, not convert to fp32
    # kv stays bf16

    # per (b, m) position
    acc_o = torch.zeros(b, s, h, d, dtype=torch.float32, device=q.device)
    sum_exp = torch.zeros(b, s, h, dtype=torch.float32, device=q.device)
    scores_max = torch.full((b, s, h), float('-inf'), dtype=torch.float32, device=q.device)

    for t in range(num_blocks):
        start = t * block
        end = min(start + block, topk)
        cur_block = end - start
        pad = block - cur_block

        # idxs: (b, m, block)
        block_idxs = topk_idxs[:, :, start:end]  # (b, m, cur_block) int32
        if pad > 0:
            block_idxs = torch.cat([block_idxs, torch.full((b, s, pad), -1, dtype=torch.int32, device=q.device)], dim=-1)

        # kv_shared: gather kv by indices, invalid -> 0
        valid_mask = (block_idxs != -1)  # (b, m, block)
        safe_idxs = block_idxs.clamp(min=0).long()  # (b, m, block)
        safe_idxs_exp = safe_idxs.unsqueeze(-1).expand(b, s, block, d)
        kv_block = kv.unsqueeze(1).expand(b, s, -1, d).gather(2, safe_idxs_exp)  # (b, m, block, d) bf16
        kv_block = kv_block * valid_mask.unsqueeze(-1).to(kv_block.dtype)  # invalid -> 0

        # acc_s init: 0 for valid, -inf for invalid
        acc_s = torch.where(
            valid_mask.unsqueeze(2).expand(b, s, h, block),
            torch.zeros(b, s, h, block, dtype=torch.float32, device=q.device),
            torch.full((b, s, h, block), float('-inf'), dtype=torch.float32, device=q.device),
        )

        # gemm: bf16 q @ bf16 kv^T, accumulate into fp32 acc_s
        # simulate tensor core: bf16 x bf16 -> fp32
        acc_s += torch.einsum("bmhd,bmkd->bmhk", q.bfloat16(), kv_block.bfloat16()).float()

        # scale
        acc_s *= softmax_scale

        # online softmax: update running max
        scores_max_prev = scores_max.clone()
        block_max = acc_s.max(dim=-1).values  # (b, m, h)
        scores_max = torch.maximum(scores_max, block_max)

        # rescale previous accumulations
        scores_scale = torch.exp(scores_max_prev - scores_max)  # (b, m, h)

        # exp scores
        acc_s = torch.exp(acc_s - scores_max.unsqueeze(-1))

        # update sum_exp
        scores_sum = acc_s.sum(dim=-1)  # (b, m, h)
        sum_exp = sum_exp * scores_scale + scores_sum

        # cast exp_scores to bf16 (matching T.copy(acc_s, acc_s_cast))
        acc_s_cast = acc_s.bfloat16()

        # rescale previous acc_o
        acc_o *= scores_scale.unsqueeze(-1)

        # gemm: bf16 exp_scores @ bf16 kv, accumulate into fp32 acc_o
        acc_o += torch.einsum("bmhk,bmkd->bmhd", acc_s_cast.bfloat16(), kv_block.bfloat16()).float()

    # add attn_sink to sum_exp
    sink = attn_sink.view(1, 1, h)  # (1, 1, h)
    sum_exp += torch.exp(sink - scores_max)

    # normalize
    acc_o /= sum_exp.unsqueeze(-1)

    # fp32 -> bf16 (matching T.copy(acc_o, o_shared) then T.copy(o_shared, o))
    o = acc_o.bfloat16()

    # strip padded heads
    if orig_h < 16:
        o = o[:, :, :orig_h, :].contiguous()

    return o
