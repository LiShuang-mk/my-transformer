# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: attention.py
# @desc:
#   This file is an pytorch version's implementation of the Transformer model referred to the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

import torch
from MyTrans.encoder import *
from MyTrans.decoder import *
from MyTrans.embedding import *


class Transformer(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        max_len,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        device,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()

        self.N = n_layers

        self.src_embedding = TransformerEmbedding(
            vocab_size=src_vocab_size, max_len=max_len, d_model=d_model, device=device
        )
        self.tgt_embedding = TransformerEmbedding(
            vocab_size=tgt_vocab_size, max_len=max_len, d_model=d_model, device=device
        )
        self.encoders = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = torch.nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
    ):
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        for i in range(self.N):
            enc_output = self.encoders[i](src_emb)
            dec_output = self.decoders[i](tgt_emb, enc_output)
        output = self.linear(dec_output)
        return torch.nn.functional.softmax(output, dim=-1)
