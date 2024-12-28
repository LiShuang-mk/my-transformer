# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: attention.py
# @desc:
#   This file is an pytorch version's implementation of the encoder referred to the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

import torch
import MyTrans.attention as attention
import MyTrans.feedforward as feedforward
import MyTrans.norm as norm


class TransformerEncoderLayer(torch.nn.Module):
    """原文中的 encoder 层，通常一个 Transformer 模型有多个 encoder 堆叠."""

    def __init__(
        self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attention.MultiHeadAttentionLayer(d_model, nhead)
        self.norm1 = norm.AddAndNorm(hidden_size=d_model, dropout_rate=dropout)

        act = None
        if activation == "relu":
            act = torch.nn.ReLU()
        elif activation == "gelu":
            act = torch.nn.GELU()
        else:
            raise ValueError("Unsupported activation function: {}".format(activation))

        self.feedforward = feedforward.FeedForwardLayer(
            d_model=d_model, d_ff=dim_feedforward, act=act
        )
        self.norm2 = norm.AddAndNorm(hidden_size=d_model, dropout_rate=dropout)

    def forward(self, embedding, padding_mask=0):
        """
        ### Args
            - embedding: 输入序列的 embedding，shape 为 [batch_size, seq_len, d_model]
            - padding_mask: 填充掩码，将因为 padding 而被设置为 padding_mask 的值置为 -inf，以让注意力机制忽略这些位置。
        """

        # self attention
        temp = self.self_attn(embedding, embedding, embedding, padding_mask)

        # sublayer 1 norm
        temp = self.norm1(embedding, temp)
        
        # feedforward
        temp = self.feedforward(temp)

        # sublayer 2 norm
        temp = self.norm2(temp, temp)

        return temp
