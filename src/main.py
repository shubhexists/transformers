import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        vocab_size: number of words in the vocabulary
        d_model: dimension of the model
        1. Creates a embedding of size d_model for each word in the vocab
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        return: (batch_size, seq_len, d_model)
        Convert the input words to their corresponding embeddings
        """
        # multiplying by sqrt(self.d_model) to scale the embeddings
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        seq_len: maximum length of the input sentence
        d_modal: dimension of the model
        dropout: dropout rate
        1. Create a matrix of shape (seq_len, d_model) with all values set to 0
        2. Create a position vector of shape (seq_len, 1) with values from 0 to seq_len-1
        3. Create a denominator vector of shape (d_model/2) with values from 0 to d_model/2-1
           and apply the formula: exp(-log(10000) * (2i/d_model))
        4. Apply the sine function to the even indices of the positional encoding matrix
           and the cosine function to the odd indices
        5. Add a batch dimension to the positional encoding matrix and register it as a buffer
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # dropout prevents overfitting of the model, randomly zeroes some values
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position_vector = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )  # (d_model/2, )

        positional_encoding[:, 0::2] = torch.sin(position_vector * denominator)
        positional_encoding[:, 1::2] = torch.cos(position_vector * denominator)

        # we unsqueeze to make it broadcastable over batch dimension (batch_size, seq_len, d_model) + (1, seq_len, d_model)
        positional_encoding.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        return: (batch_size, seq_len, d_model)
        Add positional encoding to the input embeddings
        """
        x = x + (self.positional_encoding[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, epsilon: float = 10**-6) -> None:
        """
        features: number of features for which we have to perform layer normalization, i.e, d_model
        epsilon: a very small number to prevent division by a very small number or 0
        """
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        x: (batch_size, seq_len, features)
        return: (batch_size, seq_len, features)
        Implements the layer normalization formula
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        d_model: dimension of the model. It would be the input dimension of the input layer of our feed forward network.
        d_ff: dimensions of the hidden layer. It is usually larger than the input dimensions i.e. d_model

        Architecture:
            Input (batch_size, seq_len, d_model)
                -> Linear(d_model → d_ff)
                -> ReLU (non-linearity)
                -> Dropout
                -> Linear(d_ff → d_model)
            Output (batch_size, seq_len, d_model)
        """
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.layer_2(self.dropout(torch.relu(self.layer_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float):
        """
        d_model: dimension of the model.
        head: number of parts we have to break the multihead attention block into
        """
        super().__init__()
        self.d_model = d_model
        self.heads = head
        assert d_model % head == 0, "Head should completely divide the model dimensions"

        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
