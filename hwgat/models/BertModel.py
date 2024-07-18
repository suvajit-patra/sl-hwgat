import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable
import copy


class BertModel(nn.Module):

    def __init__(self, pad_index=-1, d_model=512, nhead=8, dim_feedforward=2048, num_encoder_layers=2,
                 dropout=0.1, max_len=16) -> None:
        super().__init__()

        batch_first = True

        self.model_type = 'BertModel'
        self.d_model = d_model
        self.max_len = max_len + 1
        self.pad_index = pad_index
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout, self.max_len)

        transformer_encoder_layer = MyTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
        transformer_encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = MyTransformerEncoder(
            transformer_encoder_layer, num_encoder_layers, transformer_encoder_norm)

        self._reset_parameters()

    def make_src_mask(self, src):

        # src = [batch size, src len, feat]

        src_max = src[:, :, 0]
        src_mask = (src_max == self.pad_index)

        # src_mask = [batch size, src len]

        return src_mask

    def forward(self, src) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feat]

        Returns:
            output Tensor of shape [batch_size, nclass]
        """

        mask = None #self.make_src_mask(src)

        batch_size = src.shape[0]

        src = src * math.sqrt(self.d_model)

        cls_ = self.cls_token.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)

        src = self.pos_encoder(torch.cat((cls_ , src), dim=1))

        output = self.transformer_encoder(src, mask)  # batch, seq, d_model

        return output  # batch_size, nclass

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm) -> None:
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask) -> Tensor:
        for mod in self.layers:
            src = mod(src, src_mask=mask)

        return self.norm(src)


class MyTransformerEncoderLayer(nn.Module):
    """
    Batch First = False
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        print(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, src, src_mask) -> Tensor:
        src = self.norm1(src + self._sa_block(src, src_mask))
        src = self.norm2(src + self._ff_block(src))

        return src

    # self-attention block
    def _sa_block(self, x, attn_mask) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.relu(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)