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
        max_src_len,
        max_tgt_len,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        device,
        pad_idx=0,
        dropout=0.1,
        use_torch_version=False,
    ):
        super(Transformer, self).__init__()

        self.N = n_layers
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.src_embedding = TransformerEmbedding(
            vocab_size=src_vocab_size,
            max_len=max_src_len,
            d_model=d_model,
            device=device,
            padding_idx=pad_idx,
        )
        self.tgt_embedding = TransformerEmbedding(
            vocab_size=tgt_vocab_size,
            max_len=max_tgt_len,
            d_model=d_model,
            device=device,
            padding_idx=pad_idx,
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
                    device=device,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = torch.nn.Linear(d_model, tgt_vocab_size)

        # if use_torch_version:
        #     self.encoders = torch.nn.ModuleList(
        #         [
        #             torch.nn.TransformerEncoderLayer(
        #                 d_model=d_model,
        #                 nhead=n_heads,
        #                 dim_feedforward=d_ff,
        #                 dropout=dropout,
        #                 batch_first=True,
        #                 norm_first=True,
        #                 device=device,
        #             )
        #             for _ in range(n_layers)
        #         ]
        #     )
        #     self.decoders = torch.nn.ModuleList(
        #         [
        #             torch.nn.TransformerDecoderLayer(
        #                 d_model=d_model,
        #                 nhead=n_heads,
        #                 dim_feedforward=d_ff,
        #                 dropout=dropout,
        #                 batch_first=True,
        #                 norm_first=True,
        #                 device=device,
        #             )
        #             for _ in range(n_layers)
        #         ]
        #     )

    def forward(
        self,
        src,
        tgt,
        src_len,
        tgt_len,
    ):
        src_emb = self.src_embedding(src, src_len)
        tgt_emb = self.tgt_embedding(tgt, tgt_len)
        for i in range(self.N):
            enc_output = self.encoders[i](src_emb)
            dec_output = self.decoders[i](tgt_emb, enc_output)
        output = self.linear(dec_output)
        return output
