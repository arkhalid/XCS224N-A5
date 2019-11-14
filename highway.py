#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """Highway Networks, Srivastava et al., 2015. https://arxiv.org/abs/1505.00387"""

    def __init__(self, embed_size: int, p_drop: float):
        """ Init Highway Network.

        @param embed_size (int): Embedding size (dimensionality)
        """
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(embed_size, embed_size)
        self.w_gate = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """Takes the input and applies the highway combination to return an output with the same dimension

        @param x_conv_out (int): input to the highway network (b, embed_size)
        @returns x_word_emb (torch.Tensor): Calculated word embedding using highway net (b, embed_size)
        """
        x_proj = F.relu(self.w_proj(x_conv_out))
        x_gate = torch.sigmoid(self.w_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

### END YOUR CODE 

