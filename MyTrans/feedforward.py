# -*-- coding: utf-8 --*--
# Copyright (c) 2024 <LiShuang>
# This software is released under the MIT License.
# @author: LiShuang
# @contact: <lishuang.mk@whu.edu.cn>
# @date: 2024/12/28
# @file: attention.py
# @desc:
#   This file is an pytorch version's implementation of the FeedForward in the paper:
#       [1] Vaswani A. Attention is all you need[J]. Advances in Neural Information Processing Systems, 2017.

import torch

class FeedForwardLayer(torch.nn.Module):
    """第二个sub-layer的全连接层"""
    
    def __init__(self, d_model, d_ff, act=torch.nn.ReLU()):
        super(FeedForwardLayer, self).__init__()
        
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.activeation = act
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activeation(x)
        x = self.linear2(x)
        
        return x