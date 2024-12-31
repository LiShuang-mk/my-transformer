# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: attention.py
# @desc:
#   This file is an pytorch version's implementation of the decoder referred to the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

import torch
import MyTrans.attention as attention
import MyTrans.feedforward as feedforward
import MyTrans.norm as norm


class TransformerDecoderLayer(torch.nn.Module):
    """原文中的 decoder 层，通常一个 Transformer 模型有多个 decoder 堆叠. 但需要和 encoder 数量保持一致.."""

    def __init__(
        self,
        d_model,
        nhead,
        device,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first=True,
    ):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = attention.MultiHeadAttentionLayer(d_model, nhead)

        self.encode_attn = attention.MultiHeadAttentionLayer(d_model, nhead)

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

        self.look_ahead_mask = None
        self.device = device

        self.norm_first = norm_first

        if self.norm_first:
            self.norm1 = norm.NormOnly(hidden_size=d_model)
            self.norm2 = norm.NormOnly(hidden_size=d_model)
            self.norm_med = norm.NormOnly(hidden_size=d_model)
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.norm1 = norm.AddAndNorm(hidden_size=d_model, dropout_rate=dropout)
            self.norm2 = norm.AddAndNorm(hidden_size=d_model, dropout_rate=dropout)
            self.norm_med = norm.AddAndNorm(hidden_size=d_model, dropout_rate=dropout)
            self.dropout = None

    def forward(self, embedding, encoding, padding_mask=0):
        """
        ### Args
            - embedding: 输入序列的 embedding，shape 为 [batch_size, seq_len, d_model]
            - encoding: 编码器的输出，shape 为 [batch_size, seq_len, d_model]
            - padding_mask: 填充掩码，将因为 padding 而被设置为 padding_mask 的值置为 -inf，以让注意力机制忽略这些位置。
        """

        _, seq_len, _ = embedding.size()
        if self.look_ahead_mask is None:
            self.look_ahead_mask = (
                torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(self.device)
            )

        if self.norm_first:
            # sublayer 1 norm
            temp = self.norm1(embedding)

            # self attention
            temp = self.self_attn(
                temp, temp, temp, padding_mask, self.look_ahead_mask
            )

            # residual connection
            temp = embedding + self.dropout(temp)

            # sublayer medium norm
            temp2 = self.norm_med(temp)

            # encode attention
            temp2 = self.encode_attn(temp2, encoding, encoding, padding_mask)

            # residual connection
            temp = temp + self.dropout(temp2)

            # sublayer 2 norm
            temp2 = self.norm2(temp)

            # feedforward
            temp2 = self.feedforward(temp2)

            # residual connection
            temp = temp + self.dropout(temp2)

        else:
            # self attention
            temp = self.self_attn(
                embedding, embedding, embedding, padding_mask, self.look_ahead_mask
            )

            # sublayer 1 norm
            temp = self.norm1(embedding, temp)

            # encode attention
            temp2 = self.encode_attn(temp, encoding, encoding, padding_mask)

            # sublayer medium norm
            temp = self.norm_med(temp, temp2)

            # feedforward
            temp2 = self.feedforward(temp)

            # sublayer 2 norm
            temp = self.norm2(temp, temp2)

        return temp
