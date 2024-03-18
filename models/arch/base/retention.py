# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Copyright (c) 2023 quancs@westlake.edu.cn
# Licensed under The MIT License

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class RetNetRelPos(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, recurrent_chunk_size: int, decay: Union[int, bool, List[int], List[float]] = None):
        super().__init__()
        angle = 1.0 / (10000**torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        if decay == False:
            self.decays = [1] * num_heads
        elif isinstance(decay, Iterable):
            if isinstance(decay[0], float):
                assert decay[0] <= 1, decay
                self.decays = decay
            else:
                assert isinstance(decay[0], int) and decay[0] > 1, decay
                self.decays = [(1 - 2**(-d)) for d in decay]
        else:
            if decay is None or decay == True:
                decay = 5
            self.decays = (1 - 2**(-decay - torch.arange(num_heads, dtype=torch.float))).tolist()
        decay = torch.log(torch.tensor(self.decays, dtype=torch.float))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = recurrent_chunk_size

    def forward(self, slen: int, activate_recurrent: bool = False, chunkwise_recurrent: bool = False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)

            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

    def extra_repr(self) -> str:
        efflen = [1 / (1 - d) for d in self.decays]  # 等比数列求和
        return f"decays={self.decays} -> effective len={efflen}"

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos) -> Tensor:
    slen = x.shape[-2]
    return (x * cos[:slen]) + (rotate_every_two(x) * sin[:slen])


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class MultiScaleRetention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 2, gate_fn: str = "swish", look_ahead: int = 0, share_qk: bool = False):
        super().__init__()
        value_dim = embed_dim * value_factor
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.look_ahead = look_ahead
        self.share_qk = share_qk

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False) if share_qk == False else None
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)

        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5) if self.share_qk == False else None
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2**-1)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(self, qr: Tensor, kr: Tensor, v: Tensor, decay: Tensor, incremental_state: Dict[str, Any]):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v  # [bsz, nhead, head_dim, head_dim]
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def chunk_recurrent_forward(self, qr: Tensor, kr: Tensor, v: Tensor, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)

        tgt_len0 = tgt_len
        if tgt_len % chunk_len != 0:
            qr = torch.nn.functional.pad(qr, pad=(0, 0, 0, chunk_len - (tgt_len % chunk_len)))
            kr = torch.nn.functional.pad(kr, pad=(0, 0, 0, chunk_len - (tgt_len % chunk_len)))
            v = torch.nn.functional.pad(v, pad=(0, 0, 0, chunk_len - (tgt_len % chunk_len)))
            bsz, tgt_len, embed_dim = v.size()

        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t  # bsz * num_chunks * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v)  # bsz * num_chunks * num_heads * num_value_heads * chunk_len * head_dim

        # reduce kv in one chunk
        kv = kr_t @ (v * value_inner_decay)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * query_inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        if tgt_len0 != tgt_len:
            output = output.reshape(bsz, num_chunks * chunk_len, self.num_heads, self.head_dim)
            output = output[:, :tgt_len0]

        return output

    def forward(self, x: Tensor, rel_pos: Tensor, chunkwise_recurrent: bool = False, incremental_state: Dict[str, Any] = None, rope: bool = True) -> Tensor:
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x) if self.share_qk == False else None
        v = self.v_proj(x)
        g = self.g_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        if self.share_qk == False:
            k *= self.scaling
            k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        else:
            k = q

        qr = theta_shift(q, sin, cos) if rope else q
        kr = theta_shift(k, sin, cos) if rope else k

        if self.look_ahead > 0:
            assert incremental_state is None, "not implemented for recurrent_forward"  # recurrent_forward
            # for kr, v, pad zeros at right side; for qr, pad zeros at left side
            kr = F.pad(kr, pad=(0, 0, 0, self.look_ahead))
            v = F.pad(v, pad=(0, 0, 0, self.look_ahead))
            qr = F.pad(qr, pad=(0, 0, self.look_ahead, 0))

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        if self.look_ahead > 0:
            output = output[:, :-self.look_ahead]

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, share_qk={self.share_qk}" + (f", look_ahead={self.look_ahead}" if self.look_ahead > 0 else "")


if __name__ == '__main__':
    embed_dim, num_heads, slen, look_ahead = 96, 4, 251, 2
    ret_pos = RetNetRelPos(embed_dim=embed_dim, num_heads=num_heads, recurrent_chunk_size=100, decay=[5, 6, 7, 8])
    pp = ret_pos.forward(slen=slen + look_ahead)
    print(ret_pos)

    msret = MultiScaleRetention(embed_dim=embed_dim, num_heads=num_heads, value_factor=2, look_ahead=look_ahead, share_qk=True)
    x = torch.randn(size=(1 * 129, slen, embed_dim))
    # parallel forward
    y = msret.forward(x=x, rel_pos=pp, rope=False)
    print(y.shape)
    # chunkwise forward
    yc = msret.forward(x=x, rel_pos=ret_pos.forward(slen=slen + look_ahead, chunkwise_recurrent=True), chunkwise_recurrent=True)
    print(y.shape)
    print('equal:', torch.allclose(y, yc), (y == yc).all())
    # recurrent forward
    if look_ahead == 0:
        state = dict()
        ys = []
        for t in range(x.shape[1]):
            yr = msret.forward(x=x[:, [t], :], rel_pos=ret_pos.forward(slen=t, activate_recurrent=True), incremental_state=state)
            ys.append(yr)
        yr = torch.concat(ys, dim=1)
        print('equal:', torch.allclose(y, yr), (y == yc).all())

    # test macs
    print(msret)
    flops_forward_eval = 0
    msret.look_ahead = 0  # disable look-ahead for measuring FLOPs
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(msret, display=False) as fcm:
        ys, state = [], dict()
        for t in range(x.shape[1]):
            yr = msret.forward(x=x[:, [t], :], rel_pos=ret_pos.forward(slen=t, activate_recurrent=True), incremental_state=state)
            ys.append(yr)
        yr = torch.concat(ys, dim=1)
        loss = yr.mean()
        flops_forward_eval = fcm.get_total_flops()

    from torch.nn import MultiheadAttention
    mhsa = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
    flops_forward_eval_mhsa = 0
    with FlopCounterMode(msret, display=False) as fcm:
        y, _ = mhsa(x, x, x)
        y = y.mean()
        flops_forward_eval_mhsa = fcm.get_total_flops()
    print(flops_forward_eval / 1e9, flops_forward_eval_mhsa / 1e9)
    print(mhsa)
