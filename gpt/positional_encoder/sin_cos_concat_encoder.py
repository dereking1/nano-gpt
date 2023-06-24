import math
import torch
from torch import nn, Tensor

class SinCosConcatTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range):
        super().__init__()

        # TODO [part 2b]
        # define the encoder

        ############# YOUR CODE HERE #############
        self.encoder = nn.Embedding(n_tokens, d_model//2)
        ##########################################

        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.d_model = d_model

    def forward(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.encoder(src) * math.sqrt(self.d_model)


class SinCosConcatPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()

        # TODO [part 2b]
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it 

        ############# YOUR CODE HERE #############
        positional_encoding = torch.zeros((max_seq_len, 1, d_model//2))
        for pos in range(max_seq_len):
            for i in range(d_model//4):
                bot = math.pow(1e4, 2*i/(d_model))
                positional_encoding[pos][0][2*i] = math.sin(pos/bot)
                positional_encoding[pos][0][2*i+1] = math.cos(pos/bot)
        self.register_buffer('positional_encoding', positional_encoding)
        ##########################################

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """

        # TODO [part 2b]
        # concatenate ``positional_encoding`` to x (be careful of the shape)

        ############# YOUR CODE HERE #############
        pos_encoding = self.positional_encoding[:x.size(0)]
        pos_encoding = pos_encoding.repeat(1, x.size(1), 1)
        x = torch.cat((pos_encoding, x), dim=2)
        ##########################################

        return self.dropout(x)
