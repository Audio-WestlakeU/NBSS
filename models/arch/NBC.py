import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor


class Linear(nn.Linear):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        init.xavier_uniform_(self.weight)
        if bias:
            init.zeros_(self.bias)


class RelativePositionalEncoding(nn.Module):
    """This class returns the relative positional encoding in a range.

    for i in [-m, m]:
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    max_len : int
        Max length of the input sequences (default 1000).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = RelativePositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 239, 512])
    """

    def __init__(self, input_size, max_len=1000):
        super().__init__()
        # [-m, -m+1, ..., -1, 0, 1, ..., m]
        self.max_len = max_len
        self.zero_index = max_len
        pe = torch.zeros(self.max_len * 2 + 1, input_size, requires_grad=False)
        positions = torch.arange(-self.max_len, self.max_len + 1).unsqueeze(1).float()
        denominator = torch.exp(torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size))

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape [batch, time, feature]

        Returns:
            torch.Tensor: relative positional encoding
        """
        B, T, F = x.shape
        start, end = -T + 1, T - 1
        return self.pe[:, start + self.zero_index:end + self.zero_index + 1].clone().detach()


class RelativePositionalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(RelativePositionalMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.rel_pos = RelativePositionalEncoding(d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        init.xavier_uniform_(self.u_bias)
        init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, time_frames, feature_size = value.shape
        pos_embedding = self.rel_pos.forward(value)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))

        pos_embedding = self.pos_proj(pos_embedding)
        #### my implementation to calculate pos_score ####
        pos_index = self._get_relative_pos_index(time_frames, pos_embedding.device)
        pos_embedding = pos_embedding[:, pos_index, :].view(1, time_frames, time_frames, self.num_heads, self.d_head)
        # original matmul
        # pos_score = torch.matmul((query + self.v_bias).transpose(1, 2).unsqueeze(3), pos_embedding.permute(0, 3, 1, 4, 2)).squeeze(3)
        # faster than matmul, and saving memory
        qv_bias = (query + self.v_bias).transpose(1, 2)  # [B, N, T, D]
        pos_embedding = pos_embedding.permute(0, 3, 1, 4, 2)  # [1, N, T, D, T], 1 is broadcasted to B
        pos_score = torch.einsum("abcd,abcdf->abcf", qv_bias, pos_embedding)
        score = (content_score + pos_score) / self.sqrt_dim

        if attn_mask is not None:
            attn += attn_mask  # give -inf to mask the corresponding point

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value).transpose(1, 2)  # [B, T, N, D]

        # output
        output = output.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(output), attn

    def _get_relative_pos_index(self, T: int, device: torch.device) -> Tensor:
        with torch.no_grad():
            pos1 = torch.arange(start=0, end=T, dtype=torch.long, device=device, requires_grad=False).unsqueeze(1)
            pos2 = torch.arange(start=0, end=T, dtype=torch.long, device=device, requires_grad=False).unsqueeze(0)
            relative_pos = pos1 - pos2
            """ now, relative_pos=[
            [0,-1,-2,...,-(T-1)],
            [1, 0,-1,...,-(T-2)],
            ...
            [T-1,T-2,...,  1, 0]
            ]
            """
            pos_index = relative_pos[:, :] + (T - 1)  # (T-1) is the index of the relative position 0
        return pos_index


class NBCBlock(nn.Module):

    def __init__(
        self,
        dim_model: int = 192,
        num_head: int = 8,
        dim_ffn: int = 384,
        dropout: float = 0.1,
        activation: Callable = F.silu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        n_conv_groups: int = 384,
        conv_kernel_size: int = 3,
        conv_bias: bool = True,
        n_conv_layers: int = 3,
        conv_mid_norm: str = "GN",
    ) -> None:
        super().__init__()

        self.self_attn = RelativePositionalMultiHeadAttention(dim_model, num_head, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = Linear(dim_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_ffn, dim_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

        convs = []
        for l in range(n_conv_layers):
            convs.append(nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=conv_kernel_size, padding='same', groups=n_conv_groups, bias=conv_bias))
            if conv_mid_norm != None:
                if conv_mid_norm == 'GN':
                    convs.append(nn.GroupNorm(8, dim_ffn))
                else:
                    raise Exception('Unspoorted mid norm ' + conv_mid_norm)
            convs.append(nn.SiLU())
        self.conv = nn.Sequential(*convs)

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""

        Args:
            x: shape [batch, seq, feature]
            att_mask: the mask for attentions. shape [batch, seq, seq]

        Shape:
            out: shape [batch, seq, feature]
            attention: shape [batch, head, seq, seq]
        """

        if self.norm_first:
            x_, attn = self._sa_block(self.norm1(x), att_mask)
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x_, attn = self._sa_block(x, att_mask)
            x = self.norm1(x + x_)
            x = self.norm2(x + self._ff_block(x))

        return x, attn

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        x, attn = self.self_attn(x, attn_mask=attn_mask)
        return self.dropout1(x), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.conv(self.activation(self.linear1(x)).transpose(-1, -2)).transpose(-1, -2)))
        return self.dropout2(x)


class NBC(nn.Module):

    def __init__(
        self,
        input_size: int = 16,  # 2*8
        output_size: int = 4,  # 2*2
        n_layers: int = 4,
        encoder_kernel_size: int = 4,
        n_heads: int = 8,
        activation: Optional[str] = "",
        hidden_size: int = 192,
        norm_first: bool = True,
        ffn_size: int = 384,
        inner_conv_kernel_size: int = 3,
        inner_conv_groups: int = 8,
        inner_conv_bias: bool = True,
        inner_conv_layers: int = 3,
        inner_conv_mid_norm: str = "GN",
    ):
        super().__init__()
        # encoder
        self.encoder = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=encoder_kernel_size, stride=1)
        # self-attention net
        self.sa_layers = nn.ModuleList()
        for l in range(n_layers):
            self.sa_layers.append(
                NBCBlock(
                    dim_model=hidden_size,
                    num_head=n_heads,
                    norm_first=norm_first,
                    dim_ffn=ffn_size,
                    n_conv_groups=inner_conv_groups,
                    conv_kernel_size=inner_conv_kernel_size,
                    conv_bias=inner_conv_bias,
                    n_conv_layers=inner_conv_layers,
                    conv_mid_norm=inner_conv_mid_norm,
                ))

        # decoder
        assert activation == '', 'not implemented'
        self.decoder = nn.ConvTranspose1d(in_channels=hidden_size, out_channels=output_size, kernel_size=encoder_kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [Batch, Time, Feature]
        x = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        attns = []
        for m in self.sa_layers:
            x, attn = m(x)
            attns.append(attn)
        y = self.decoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y.contiguous()  # , attns


if __name__ == '__main__':
    Batch, Freq, Time, Chn, Spk = 1, 257, 100, 8, 2
    x = torch.randn((Batch * Freq, Time, Chn * 2))
    m = NBC(input_size=Chn * 2, output_size=Spk * 2, n_layers=4)
    y = m(x)
    print(y.shape)
