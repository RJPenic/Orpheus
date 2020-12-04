import torch
from torch import nn
import torch.nn.functional as F

import dataset as dl

class HAN_Model(nn.Module):
    def __init__(self,
            vocab,
            input_dim = 300,
            word_hidden_dim = 50,
            line_hidden_dim = 50,
            word_att_layers = 1,
            line_att_layers = 1,
            word_droput = 0.5,
            line_dropout = 0.5,
            pretrain_file = None,
            num_classes = 10):
        super().__init__()
        
        self.word_att = Attention(input_dim, word_hidden_dim, word_att_layers, word_droput)
        self.line_att = Attention(2 * word_hidden_dim, line_hidden_dim, line_att_layers, line_dropout)

        self.final_proj = nn.Linear(2 * line_hidden_dim, num_classes)

        self.encoder = dl.load_vec_repr(vocab, file = pretrain_file, freeze = pretrain_file is not None)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.encoder(x)

        temp = []

        for line in x:
            temp.append(self.word_att(line))

        x = torch.cat(temp, 0)
        x = self.line_att(x.permute(1, 0, 2))

        x = self.final_proj(x)

        return x

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, batch_first = True):
        super().__init__()

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers = num_layers, bidirectional = True, batch_first = batch_first)
        self.dropout = nn.Dropout(p = dropout)

        self.att_lin = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.relevance_lin = nn.Linear(2 * hidden_dim, 1, bias = False)

    def forward(self, x):
        x, _ = self.gru(x)

        alphas = self.att_lin(x)
        alphas = torch.tanh(alphas)
        alphas = self.relevance_lin(x)
        alphas = F.softmax(alphas, dim = 1)

        x = torch.sum(x * alphas, dim = 1).unsqueeze(0)
        x = self.dropout(x)

        return x
