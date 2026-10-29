# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2018. Github developer `pengshuang` for ScaledDotProductAttention implementation. 

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import math
import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False, align_to="input"):
        super(MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.align_to = align_to
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            if align_to == "output":
                self.W_res = nn.Linear(input_dim, self.output_dim, bias=False)
            elif align_to == "input":
                self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query
        
        # linear projection
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        # concat heads
        output = output.view(batch_size, -1, self.output_dim)
        # final linear projection
        if self.W_res is not None:
            if self.align_to == "output": # AutoInt style
                residual = self.W_res(residual)
            elif self.align_to == "input": # Transformer stype
                output = self.W_res(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output, attention


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class TargetQueryAttention(nn.Module):
    """Single-query scaled dot-product attention from BiSID-CTR.

    Attn(q, Z) = sum_j alpha_j z_j,  alpha = softmax(Z W_k q / sqrt(d)).
    Query length is 1, so the cost is O(nd).
    """

    def __init__(self, dim, num_heads=4):
        super(TargetQueryAttention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_k = nn.Linear(dim, dim, bias=False)

    def forward(self, query, keys, mask=None):
        batch_size, seq_len, _ = keys.size()
        q = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(keys).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = attn * mask.unsqueeze(1).unsqueeze(2).float()
        output = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, self.dim)
        if mask is not None:
            nonempty = mask.any(dim=-1, keepdim=True).float()
            output = output * nonempty
        return output


class DINAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_units=(64,), hidden_activations="ReLU", dropout_rate=0.0, use_softmax=False):
        super(DINAttention, self).__init__()
        self.use_softmax = use_softmax
        layers = []
        input_dim = embedding_dim * 4
        hidden_units = list(hidden_units)
        for hidden in hidden_units:
            layers.append(nn.Linear(input_dim, hidden))
            if str(hidden_activations).lower() == "dice":
                from .activation import Dice
                layers.append(Dice(hidden))
            else:
                layers.append(nn.ReLU() if str(hidden_activations).lower() == "relu" else nn.PReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, target, sequence, mask=None):
        seq_len = sequence.size(1)
        target_expand = target.unsqueeze(1).expand(-1, seq_len, -1)
        din_input = torch.cat([
            target_expand,
            sequence,
            target_expand - sequence,
            target_expand * sequence,
        ], dim=-1)
        scores = self.mlp(din_input).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9 if self.use_softmax else 0.0)
        if self.use_softmax:
            weights = torch.softmax(scores, dim=-1)
            if mask is not None:
                weights = torch.nan_to_num(weights, nan=0.0)
        else:
            weights = torch.sigmoid(scores)
            if mask is not None:
                weights = weights * mask.float()
        return torch.sum(weights.unsqueeze(-1) * sequence, dim=1)


