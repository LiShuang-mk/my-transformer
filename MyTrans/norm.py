# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: norm.py
# @desc:
#   This file is an pytorch version's implementation of the layer normalization of the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.
#       [2] Lei Ba J, Kiros J R, Hinton G E. Layer normalization[J]. ArXiv e-prints, 2016: arXiv: 1607.06450.

import torch


class LayerNormalization(torch.nn.Module):
    """横向归一化，先将一个神经层上的所有神经元归一化为均值为0，方差为1的分布，\
        然后再将归一化后的结果应用可学习的线性变化（缩放因子和偏置），最后得到输出。
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.eps) + self.bias
    
    
class AddAndNorm(torch.nn.Module):
    """残差连接+层归一化，即先将两个神经层的输出相加，然后再进行层归一化。\
        层归一化前先进行 dropout，增强泛化。
    """
    def __init__(self, hidden_size, dropout_rate=0.1, eps=1e-6):
        super(AddAndNorm, self).__init__()
        
        self.layer_norm = LayerNormalization(hidden_size, eps)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # drop out
        y = self.dropout(y)
        
        # residual connection
        res = x + y
        
        # layer normalization
        res = self.layer_norm(res)
        
        return res
    
class NormOnly(torch.nn.Module):
    
    def __init__(self, hidden_size, eps=1e-6):
        super(NormOnly, self).__init__()
        
        self.layer_norm = LayerNormalization(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # layer normalization
        x = self.layer_norm(x)
        
        return x
