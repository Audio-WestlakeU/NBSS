from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn.common_types import _size_1_t

from models.arch.base.linear_group import LinearGroup
from models.arch.base.non_linear import *
from models.arch.base.norm import *
from models.arch.base.retention import MultiScaleRetention, RetNetRelPos

try:
    from mamba_ssm import Mamba
    from mamba_ssm.utils.generation import InferenceParams
except:
    Mamba = None


class CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        look_ahead: int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.look_ahead = look_ahead
        assert look_ahead <= self.kernel_size[0] - 1, (look_ahead, self.kernel_size)

    def forward(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        # x [B,H,T]
        # state[name]存在，说明可以使用state里面的内容进行padding
        B, H, T = x.shape
        if state is None or id(self) not in state:
            x = F.pad(x, pad=(self.kernel_size[0] - 1 - self.look_ahead, self.look_ahead))
        else:
            x = torch.concat([state[id(self)], x], dim=-1)
        if state is not None:
            state[id(self)] = x[..., -self.kernel_size + 1:]
        x = super().forward(x)
        return x

    def extra_repr(self):
        if self.look_ahead == 0:
            return super().extra_repr()
        else:
            return super().extra_repr() + f", look ahead={self.look_ahead}"


class SpatialNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
            padding: str = 'zeros',
            full: nn.Module = None,
            attention: str = 'mhsa',
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        if attention.startswith('ret'):  # e.g. ret(1,share_qk)
            attn_params = attention[4:-1].split(',')
            assert attn_params[1] in ['share_qk', 'not_share_qk'], attn_params
            value_factor = int(attn_params[0])
            self.mhsa = MultiScaleRetention(embed_dim=dim_hidden, num_heads=num_heads, value_factor=value_factor, share_qk=attn_params[1] == 'share_qk')
        elif attention.startswith('mamba'):  # e.g. mamba(16,4)
            attn_params = attention[6:-1].split(',')
            d_state, mamba_conv_kernel = int(attn_params[0]), int(attn_params[1])
            self.mhsa = Mamba(d_model=dim_hidden, d_state=d_state, d_conv=mamba_conv_kernel, layer_idx=0)
        else:
            self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.attention = attention
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        if attention.startswith('mamba') and 'not_replace_ffn' not in attention:
            self.norm_tconvffn = new_norm(norms[1], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
            self.tconvffn = Mamba(d_model=dim_hidden, d_state=d_state, d_conv=mamba_conv_kernel, layer_idx=0)
        else:
            self.tconvffn = nn.ModuleList([
                new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
                nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
                nn.SiLU(),
                CausalConv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, groups=t_conv_groups),
                nn.SiLU(),
                CausalConv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, groups=t_conv_groups),
                new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
                nn.SiLU(),
                CausalConv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, groups=t_conv_groups),
                nn.SiLU(),
                nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
            ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None, chunkwise_recurrent: bool = True, rope: bool = True, state: Dict[int, Any] = None, inference: bool = False) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        attn = None
        if isinstance(self.mhsa, Mamba):
            x = x + self._mamba(x, self.mhsa, self.norm_mhsa, self.dropout_mhsa, inference)
        else:
            x_, attn = self._tsa(x, att_mask, chunkwise_recurrent, rope, state=state, inference=inference)
            x = x + x_
        if isinstance(self.tconvffn, Mamba):
            x = x + self._mamba(x, self.tconvffn, self.norm_tconvffn, self.dropout_tconvffn, inference)
        else:
            x = x + self._tconvffn(x, state=state)
        return x, attn

    def _mamba(self, x: Tensor, mamba: Mamba, norm: nn.Module, dropout: nn.Module, inference: bool = False):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)
        if inference:
            inference_params = InferenceParams(T, B * F)
            xs = []
            for i in range(T):
                inference_params.seqlen_offset = i
                xi = mamba.forward(x[:, [i], :], inference_params)
                xs.append(xi)
            x = torch.concat(xs, dim=1)
        else:
            x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor], chunkwise_recurrent: bool, rope: bool = True, state: Dict[int, Any] = None, inference: bool = False) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        if isinstance(self.mhsa, MultiheadAttention):
            need_weights = False if hasattr(self, "need_weights") else self.need_weights
            # seems MHSA for long utterance inference has this issue https://github.com/pytorch/pytorch/issues/120790
            x, attn = self.mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=attn_mask, is_causal=True)
        else:
            if inference == False:
                x = self.mhsa.forward(x, rel_pos=attn_mask, incremental_state=state, chunkwise_recurrent=chunkwise_recurrent, rope=rope)
            else:
                xs, state = [], dict()
                for i in range(T):
                    xi = self.mhsa.forward(x[:, [i], :], rel_pos=attn_mask[i], incremental_state=state)
                    xs.append(xi)
                x = torch.concat(xs, dim=1)
            attn = None
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if isinstance(m, CausalConv1d):
                x = m(x, state=state)
            elif isinstance(m, nn.GroupNorm) or "GroupNorm" in type(m).__name__:  # normalize along H & F
                x = x.reshape(B, F, -1, T).transpose(1, -1).reshape(B * T, -1, F)
                x = m(x)
                x = x.reshape(B, T, -1, F).transpose(1, -1).reshape(B * F, -1, T)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class OnlineSpatialNet(nn.Module):

    def __init__(
        self,
        dim_input: int,  # the input dim for each time-frequency point
        dim_output: int,  # the output dim for each time-frequency point
        num_layers: int,
        dim_squeeze: int,
        num_freqs: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_heads: int = 2,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = 'zeros',
        full_share: int = 0,  # share from layer 0
        attention: str = 'mhsa(251)',  # mhsa(frames), ret(factor)
        decay: Union[int, bool, List[int], List[float]] = 5,
        chunkwise_recurrent: bool = True,
        rope: Union[bool, str] = False,
    ):
        super().__init__()
        assert attention.startswith('mhsa') or attention.startswith('ret') or attention.startswith('mamba'), attention
        assert attention.startswith('mhsa') or attention.startswith('mamba') or attention in [
            'mhsa(inf)', 'mhsa(501)', 'mhsa(251)', 'mhsa(188)', 'mhsa(126)', 'ret(2)', 'ret(2,share_qk)', 'ret(2,not_share_qk)'
        ], attention
        assert rope in [True, False, 'ALiBi'], rope
        if attention == 'ret(2)':  # 兼容之前训练的版本，在不使用旋转位置编码的时候，共享Q/K
            attention = 'ret(2,share_qk)' if rope == False else 'ret(2,not_share_qk)'

        self.num_heads = num_heads
        self.chunkwise_recurrent = chunkwise_recurrent
        self.pos = None
        if attention.startswith('ret'):
            self.pos = RetNetRelPos(embed_dim=dim_hidden, num_heads=num_heads, recurrent_chunk_size=64, decay=decay)
        elif attention.startswith('mamba'):
            self.attn_scope = 1
        else:
            import math
            self.attn_scope = int(attention[5:-1]) if attention[5:-1] != 'inf' else math.inf
        self.rope = rope

        # encoder
        self.encoder = CausalConv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, look_ahead=0)

        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                attention=attention,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor, inference: bool = False, return_attn_score: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]

        chunkwise_recurrent = True if inference == False else self.chunkwise_recurrent
        mask = self.get_causal_mask(slen=T, device=x.device, chunkwise_recurrent=chunkwise_recurrent, batch_size=B, inference=inference)

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        for i, m in enumerate(self.layers):
            setattr(m, "need_weights", return_attn_score)
            x, attn = m(x, mask, chunkwise_recurrent, self.rope, None, inference)
            if return_attn_score:
                attns.append(attn)

        y = self.decoder(x)
        if return_attn_score:
            return y.contiguous(), attns
        else:
            return y.contiguous()

    def get_causal_mask(self, slen: int, device=None, chunkwise_recurrent: bool = True, batch_size: int = None, inference: bool = False):
        if isinstance(self.pos, RetNetRelPos):
            if inference == False:
                mask = self.pos.forward(slen=slen, chunkwise_recurrent=chunkwise_recurrent)
            else:
                mask = []
                for t in range(slen):
                    rel_pos = self.pos.forward(slen=t, activate_recurrent=True)
                    mask.append(rel_pos)
        else:
            pos1 = torch.arange(start=0, end=slen, dtype=torch.long, device=device, requires_grad=False).unsqueeze(1)
            pos2 = torch.arange(start=0, end=slen, dtype=torch.long, device=device, requires_grad=False).unsqueeze(0)
            relative_pos = pos1 - pos2
            """ now, relative_pos=[
            [0,-1,-2,...,-(T-1)],
            [1, 0,-1,...,-(T-2)],
            ...
            [T-1,T-2,...,  1, 0]
            ]
            """
            if self.rope == 'ALiBi':
                assert batch_size is not None, batch_size
                m = (2.0**(-8 / torch.arange(1, self.num_heads + 1, 1, device=device))).reshape(self.num_heads, 1, 1)
                m = torch.concat([m] * batch_size, dim=0)
                relative_pos = torch.where((relative_pos >= 0) * (relative_pos < self.attn_scope), relative_pos.abs() * -1, -torch.inf)
                mask = m * relative_pos
                return mask

            mask = torch.where((relative_pos >= 0) * (relative_pos < self.attn_scope), 0.0, -torch.inf)
        return mask


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.OnlineSpatialNet
    model = OnlineSpatialNet(
        dim_input=12,
        dim_output=4,
        num_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        num_heads=4,
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=["LN", "LN", "GN", "LN", "LN", "LN"],
        dim_squeeze=8,
        num_freqs=129,
        full_share=0,
        attention='mamba(16,4)',
        rope=False,
    ).cuda()
    print(model)

    x = torch.randn((1, 129, 251, 12)).cuda() # 6-channel, 4s, 8 kHz
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(model, display=False) as fcm:
        res = model(x, inference=True).mean()
        flops_forward_eval = fcm.get_total_flops()
    for k, v in fcm.get_flop_counts().items():
        ss = f"{k}: {{"
        for kk, vv in v.items():
            ss += f" {str(kk)}:{vv}"
        ss += " }"
        print(ss)
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval/4e9:.2f}G/s, params={params_eval/1e6:.2f} M")

    # check if the implementation is causal or not
    x = torch.randn((1, 129, 1024, 12)).cuda()
    y1024 = model(x)
    y1000 = model(x[:, :, :1000, :])
    print('causal:', (y1024[:, :, :1000, :] == y1000).all().item())
