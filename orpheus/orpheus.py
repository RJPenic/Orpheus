import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import LyricsDataset, pad_collate_fn
from han_model import HAN_Model
from argparse import ArgumentParser

from pytorch_lightning.metrics import F1


class Orpheus(pl.LightningModule):
    def __init__(self, args, dataset, model: nn.Module = None):
        super().__init__()
        self.args = args
        self.model = model
        self.dataset = dataset

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        self.train_f1 = F1(args.num_classes)
        self.val_f1 = F1(args.num_classes)
        self.test_f1 = F1(args.num_classes)

    # load/download and prepare data, called only once
    def prepare_data(self):
        if self.args.dataset is None:
            print("Program requires 1 command line argument:")
            print("  --dataset <path to directory containing dataset>")
            sys.exit(1)

    # split into training and validation dataset
    def setup(self, stage: str):
        n_train = int(len(dataset) * 0.8)
        n_val = int(len(dataset) * 0.1)
        n_test = len(dataset) - n_val - n_train
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = True, batch_size=args.batch, pin_memory=True, collate_fn=pad_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle = False, batch_size=args.batch, pin_memory=True, collate_fn=pad_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle = False, batch_size=args.batch, pin_memory=True, collate_fn=pad_collate_fn)

    def training_step(self, batch, batch_idx):
        x, targets = batch
        x = self.model(x)

        loss = F.cross_entropy(x, targets)
        self.log('train_loss', loss)

        self.log('train_acc', self.train_accuracy(x, targets))
        self.log('train_f1', self.train_f1(x, targets))

        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        x = self.model(x)

        loss = F.cross_entropy(x, targets)
        self.log('val_loss', loss)

        self.log('val_acc', self.val_accuracy(x, targets))
        self.log('val_f1', self.val_f1(x, targets))

        return loss

    def test_step(self, batch, batch_idx):
        x, targets = batch
        x = self.model(x)

        loss = F.cross_entropy(x, targets)
        self.log('test_loss', loss)

        self.log('test_acc', self.test_accuracy(x, targets))
        self.log('test_f1', self.test_f1(x, targets))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents = [parent_parser], add_help = False)

        parser.add_argument('--batch', type=int, default = 16,
                        help = "Size of a single batch")
        parser.add_argument('--dataset', type=str,
                    	help = "Path to file containing dataset")
        parser.add_argument('--lr', type=float, default = 3e-4,
                        help = "Learning rate")
        parser.add_argument('--num_workers', type=int, default=4,
                        help="How many subprocesses to use for data loading")
        parser.add_argument('--del_stopwords', default=False, action='store_true',
                        help="Remove stopwords in dataset")

        return parser


if __name__ == "__main__":
    parser = ArgumentParser()

    parser = HAN_Model.add_model_specific_args(parser)
    parser = Orpheus.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    dataset = LyricsDataset.from_file(args.dataset, remove_stop_words = args.del_stopwords)
    model = Orpheus(args, dataset, HAN_Model(dataset.text_vocab,
                                        input_dim = args.input_dim,
                                        word_hidden_dim = args.word_hidden_dim,
                                        line_hidden_dim = args.line_hidden_dim,
                                        word_att_layers = args.word_att_layers,
                                        line_att_layers = args.line_att_layers,
                                        word_dropout = args.word_dropout,
                                        line_dropout = args.line_dropout,
                                        pretrain_file = args.pretrain_file,
                                        num_classes = args.num_classes))

    trainer = pl.Trainer(fast_dev_run = args.fast_dev_run, benchmark = args.benchmark,
                        gpus = args.gpus, max_epochs = args.max_epochs, gradient_clip_val=1.0)
    trainer.fit(model)
    result = trainer.test()
    print(result)
