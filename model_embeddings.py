#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        char_emb_dim = 50
        max_sentence_len = 21
        p_drop = 0.3
        kernel_size = 5
        pad_token_idx = vocab.char2id['<pad>']
        self.char_emb = nn.Embedding(len(vocab.char2id), char_emb_dim, pad_token_idx)
        self.cnn = CNN(kernel_size, embed_size, char_emb_dim, max_sentence_len)
        self.hwy = Highway(embed_size, p_drop)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        # shape (sentence_length, batch_size, max_word_length, char_emb_dim)
        char_embedded = self.char_emb(input)
        sentence_length, batch_size, max_word_length, char_emb_dim = list(char_embedded.size())
        # shape (sentence_length, batch_size, char_emb_dim, max_word_length)
        char_embedded_permuted = char_embedded.permute(0, 1, 3, 2)
        # shape (sentence_length * batch_size, char_emb_dim, max_word_length)
        char_embedded_permuted_flattened = char_embedded_permuted.reshape(sentence_length * batch_size,
                                                                          char_emb_dim, max_word_length)
        # shape(sentence_length * batch_size, word_emb_dim)
        word_emb_flattened = self.cnn(char_embedded_permuted_flattened)
        # shape(sentence_length * batch_size, word_emb_dim)
        word_emb_flattened = self.hwy(word_emb_flattened)
        # shape(sentence_length, batch_size, word_emb_dim)
        word_embedded = word_emb_flattened.reshape(sentence_length, batch_size, -1)
        return word_embedded
        ### END YOUR CODE

