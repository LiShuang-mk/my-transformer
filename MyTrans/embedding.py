# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: attention.py
# @desc:
#   This file is an pytorch version's implementation of the embedding with positional encoding referred to the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

import torch


class PositionalEncoding(torch.nn.Module):
    """正弦波位置编码"""

    def __init__(self, d_model, max_len, device):
        """
        ### Args
            - d_model: 整个模型的嵌入维度
            - max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        ### Args
            - x: (batch_size, seq_len)
        ### Returns
            - (seq_len, d_model)
        """
        _, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(torch.nn.Module):
    """Transformer Embedding"""

    def __init__(
        self, vocab_size, max_len, d_model, device, padding_idx=0, dropout=0.1
    ):
        """ """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        ### Args
            - x: (batch_size, seq_len)
        ### Returns
            - (batch_size, seq_len, d_model)
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        emb = tok_emb[:] + pos_emb
        return self.drop_out(emb)
