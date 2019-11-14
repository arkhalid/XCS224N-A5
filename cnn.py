#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i


class CNN(nn.Module):
    """CNN model for char based encoder"""

    def __init__(self, kernel_size: int, num_filters: int, char_embed_dim: int, max_word_len: int):
        """
        Init CNN.
        @param kernel_size (int): window size for the 1D convolution
        @param num_filters (int): number of filters for 1D convolution
        @param char_embed_dim (int): character embedding dimension
        @param embed_size (int): dim of char embeddings
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embed_dim, num_filters, kernel_size)
        self.max_pool = nn.MaxPool1d(max_word_len)

    def forward(self, char_embedded_word: torch.Tensor) -> torch.Tensor:
        """
        Takes the word embedded through char embeddings, and passes through a convnet to generate the word embeddings
        @param char_embedded_word (torch.Tensor): word embedded through char embeddings (batch_size, char_emb_dim, max_word_len)
        @returns word_embedding (torch.Tensor): word embedded using convnet (batch_size, word_emb_dim)
        """
        x_conv = self.conv(char_embedded_word)
        x_conv_out = F.relu(x_conv)
        word_embedding = self.max_pool(x_conv_out)
        word_embedding = word_embedding.squeeze(2)
        return word_embedding

### END YOUR CODE