import torch
from torch import nn
import torch.nn.functional as F

import dataset as dl
from argparse import ArgumentParser
from orpheus import Orpheus

class HAN_Model(Orpheus):
    def __init__(self, vocab, args):
        super().__init__(args)
        self.encoder = dl.load_vec_repr(vocab, d = args.input_dim, file = args.pretrain_file, freeze = args.pretrain_file is not None)
        self.word_att = Attention(args.input_dim, args.word_hidden_dim, args.word_att_layers, args.word_dropout)
        self.line_att = Attention(2 * args.word_hidden_dim, args.line_hidden_dim, args.line_att_layers, args.line_dropout)

        self.final_proj = nn.Linear(2 * args.line_hidden_dim, args.num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents = [parent_parser], add_help = False)

        parser.add_argument('--pretrain_file', type=str, default = None,
                        help="Path to file containing word vector representations")
        parser.add_argument('--word_hidden_dim', type=int, default = 50,
                        help="Size of hidden states in word attention module")
        parser.add_argument('--line_hidden_dim', type=int, default = 50,
                        help="Size of hidden states in line attention module")
        parser.add_argument('--word_att_layers', type=int, default = 1,
                        help="Number of layers used in word attention module")
        parser.add_argument('--line_att_layers', type=int, default = 1,
                        help="Number of layers used in line attention module")
        parser.add_argument('--word_dropout', type=float, default = 0.5,
                        help="Probability of dropout on the output of word attention module")
        parser.add_argument('--line_dropout', type=float, default = 0.5,
                        help="Probability of dropout on the output of line attention module")
        parser.add_argument('--input_dim', type=int, default = 100,
                        help="Dimension of model inputs")

        return parser

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.encoder(x)

        temp = []

        for line in x:
            temp.append(self.word_att(line))

        x = torch.cat(temp, 1)
        x = self.line_att(x)
        x = x.squeeze(1)

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

        x = torch.sum(x * alphas, dim = 1).unsqueeze(1)
        x = self.dropout(x)

        return x
