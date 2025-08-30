# In pytorch, forward function of each class is called automatically, so we do not need to call it each time we call that class.

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
        positional_encoding = positional_encoding.unsqueeze(0)  # (1, seq_len, d_model)
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
                -> Linear(d_ff → d_mudrodip?tab=overview&from=2025-08-01&to=2025-08-29odel)
            Output (batch_size, seq_len, d_model)
        """
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.layer_2(self.dropout(torch.relu(self.layer_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        """
        d_model: dimension of the model.
        head: number of parts we have to break the multihead attention block into
        Initialize four linear layers of size d_model by d_model which we will use later
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

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        query, key and value are the input matrices to calculate the attention
        mask is used in a case where we need to ignore the interactions between certain values.
        For eg. While using this in a decoder, we would mask all the keys ahead of the word.
        Similarly, we will ignore all the padded elements in a sentence.

        This function implements the the attention calculation logic.
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # "@" represents matrix multiplication in pytorch

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float("-inf"))
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, query, key, value, mask):
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # We now divide the matrices in `heads` part.
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, head, (d_model // head)) --> (batch_size, head, seq_len, (d_model // head))
        query = query.view(
            query.shape[0], query.shape[1], self.heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.heads, self.d_k
        ).transpose(1, 2)

        # Calculate the attention values and the final output after multiplying it with `value`
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # (batch_size, head, seq_len, (d_model // head)) --> (batch_size, seq_len, head, (d_model // head)) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        """
        This class is basically a wrapper around all the blocks that we'll use in the transformer.
        It will pass through that layer and automatically apply dropout and layer normalization to prevent values to go out of bound.


        [LayerNorm -> Sublayer -> Dropout] + Input
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features=features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """
        This defines the structure of the encoder block.
        First is the multihead self attention block and the second is the feed forward block
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        This is the main Encoder class built up of multiple "EncoderBlock" classes
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_layer: FeedForwardBlock,
        features: int,
        dropout: float,
    ) -> None:
        """
        This class defines the structure of the decoder block.
        First is the masked multihead self attention layer which takes in the target embeddings,
        Second is the cross multihead attention layer which takes query from the decoder but key and value from the encoder
        Thirdly the feed forward layer that takes the output of the cross multi head attention
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, target_mask, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_layer)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int) -> None:
        """ """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x, encoder_output, target_mask, src_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, src_mask)
        self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbeddings,
        target_embedding: InputEmbeddings,
        src_position: PositionalEncoding,
        target_position: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_position = src_position
        self.target_position = target_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, target_mask, src_mask)

    def projection_layer(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    head: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    src_vocab_size: number of words in the vocab
    target_vocab_size: its the output of the target vocab
    src_seq_len: it represent the maxium number of words in a sentence
    """
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    target_embeddings = InputEmbeddings(d_model, target_vocab_size)

    src_positional_embeddings = PositionalEncoding(d_model, src_seq_len, dropout)
    target_postional_embeddings = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for i in range(N):
        encoder_self_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model, head, dropout
        )
        feed_forward_layer = MultiHeadAttentionBlock(d_model, head, dropout)
        encoder_blocks.append(
            EncoderBlock(
                d_model, encoder_self_multi_head_attention_block, feed_forward_layer
            )
        )

    decoder_blocks = []
    for i in range(N):
        decoder_masked_multi_head_attention_block = MultiHeadAttentionBlock(
            d_model, head, dropout
        )
        cross_multihead_attention_block = MultiHeadAttentionBlock(
            d_model, head, dropout
        )
        feed_forward_layer = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(
            DecoderBlock(
                decoder_masked_multi_head_attention_block,
                cross_multihead_attention_block,
                feed_forward_layer,
                d_model,
                dropout,
            )
        )

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        src_embeddings,
        target_embeddings,
        src_positional_embeddings,
        target_postional_embeddings,
        projection_layer,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
