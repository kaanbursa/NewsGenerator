import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, n_token, n_input, n_head, n_hidden, n_layers, vocab, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'
        self.src_mask = None

        self.pos_enc = PositionalEncoder(n_input,dropout)
        encoder_layers = TransformerEncoderLayer(n_input,n_head,n_hidden,dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,n_layers)
        self.encoder = nn.Embedding(n_token,n_input)
        self.encoder.weight.data.copy_(vocab.vectors)
        self.n_input = n_input
        self.decoder = nn.Linear(n_input,n_token)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.n_input)
        src = self.pos_enc(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoder(nn.Module):

    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoder,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
