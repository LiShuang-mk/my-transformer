# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: attention.py
# @desc:
#   This file is an pytorch version's implementation of the attention mechanism referred to the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

import torch


class ScaledDotProductAttention(torch.nn.Module):
    """原文中的 Scaled Dot-Product Attention (SDP-Attention)"""

    def __init__(self, d_k: int, d_v: int):
        """初始化一个 SDP-Attention

        ### Args
            - d_k (int): \
                Attention 机制内部向量维度 (d_q, d_k)
            - d_v (int): \
                Attention 机制输出向量维度 (d_v)
        """
        super(ScaledDotProductAttention, self).__init__()

        self.temper = d_k**0.5

    def forward(self, q, k, v, mask=0, look_ahead_mask=None):
        """计算 SDP-Attention

        Args:
            q : (b, n, d_k)
            k : (b, n, d_k)
            v : (b, n, d_v)
            mask (int, optional): 需要被 mask 的值，等于 mask 的位置的 Attention 权重被置为 -inf. 默认是 None.

        Returns:
            tensor: (b, n, d_v)
        """

        # Q * K^T / sqrt(d_k)
        attn: torch.Tensor = torch.bmm(q, k.transpose(1, 2)) / self.temper

        # mask padding to -inf
        attn[attn[:, :] == mask] = float("-inf")
        
        # prevent look-ahead to future words
        if look_ahead_mask is not None:
            attn.masked_fill(look_ahead_mask == 0, float("-inf"))

        # softmax(·)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # multiple to v
        attn = torch.bmm(attn, v)

        return attn


class AttentionHead(torch.nn.Module):
    """原文中的 Attention Head"""

    def __init__(self, d_model: int, d_k: int, d_v: int):
        """初始化一个 Attention Head

        ### Args
            - d_model (int): \
                整个模型中嵌入后的向量维度
            - d_k (int): \
                Attention 机制内部向量维度 (d_q, d_k)
            - d_v (int): \
                Attention 机制输出向量维度 (d_v)
        """
        super(AttentionHead, self).__init__()

        self.linear_q = torch.nn.Linear(d_model, d_k)  # W^Q
        self.linear_k = torch.nn.Linear(d_model, d_k)  # W^K
        self.linear_v = torch.nn.Linear(d_model, d_v)  # W^V

        self.attention = ScaledDotProductAttention(d_k, d_v)

    def forward(self, q, k, v, mask=0, look_ahead_mask=None):
        # linear projection
        q = self.linear_q(q)  # (b, n, d_k)
        k = self.linear_k(k)  # (b, n, d_k)
        v = self.linear_v(v)  # (b, n, d_v)

        # attention
        attn = self.attention(q, k, v, mask, look_ahead_mask)

        return attn


class MultiHeadAttentionLayer(torch.nn.Module):
    """原文中的 Multi-Head Attention (MH-Attention)"""

    def __init__(self, d_model: int, num_heads: int):
        """初始化一个 MH-Attention

        ### Args
            - d_model (int): \
                整个模型中嵌入后的向量维度
            - num_heads (int): \
                多头注意力的头数，d_model/num_heads 应该是整数，\
                这个就是model中每一个head的向量维度
        """
        super(MultiHeadAttentionLayer, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 应该被 num_heads ({num_heads}) 整除")

        self.d_model = d_model
        self.h = num_heads

        self.d_k = d_model // self.h
        self.d_v = d_model // self.h

        self.heads = torch.nn.ModuleList(
            [AttentionHead(d_model, self.d_k, self.d_v) for _ in range(self.h)]
        )

        self.linear_final = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=0, look_ahead_mask=None):
        # multi-head attention
        output = torch.cat(
            [head(q, k, v, mask, look_ahead_mask) for head in self.heads], dim=-1
        )  # (b, n, d_model)

        # linear projection
        output = self.linear_final(output)

        return output
