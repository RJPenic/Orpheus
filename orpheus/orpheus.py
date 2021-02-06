import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sklearn.metrics import confusion_matrix
import numpy as np

from dataset import LyricsDataset, pad_collate_fn
import argparse

class TrainingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = LyricsDataset.from_file(args.data_path, labels = ['pop', 'hip-hop', 'rock', 'metal', 'country', 'jazz'], take_rates = [1.0, 1.0, 0.5, 1.0, 1.0, 1.0])

        total_ratio_sum = self.args.data_ratio[0] + self.args.data_ratio[1] + self.args.data_ratio[2]
        n_train = int(len(self.dataset) * (self.args.data_ratio[0] / total_ratio_sum))
        n_val = int(len(self.dataset) * (self.args.data_ratio[1] / total_ratio_sum))
        n_test = len(self.dataset) - n_val - n_train

        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(self.dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.train_data, self.args.train_bs, shuffle=True, 
                num_workers=self.args.train_workers, pin_memory=True, collate_fn=pad_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.args.val_bs, 
                num_workers=self.args.val_workers, pin_memory=True, collate_fn=pad_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.args.test_bs, 
                num_workers=self.args.test_workers, pin_memory=True, collate_fn=pad_collate_fn)

    @staticmethod
    def add_data_module_specific_args(parents):
        parser = argparse.ArgumentParser(parents=[parents], add_help=False)

        # Dataset path
        parser.add_argument('data_path', type=str,
                        help="Path to dataset file")

        # Train dataloader args
        parser.add_argument('--train_bs', type=int, default=16,
                        help="Train mini batch size")
        parser.add_argument('--train_workers', type=int, default=0,
                        help="Number of train dataloader worker processes")

        # Validation dataloader args
        parser.add_argument('--val_bs', type=int, default=16,
                        help="Validation mini batch size")
        parser.add_argument('--val_workers', type=int, default=0,
                        help="Number of validation dataloader worker processes")

        # Test dataloader args
        parser.add_argument('--test_bs', type=int, default=16,
                        help="Test mini batch size")
        parser.add_argument('--test_workers', type=int, default=0,
                        help="Number of test dataloader worker processes")

        parser.add_argument('--data_ratio', type=float, nargs = 3, default=[0.7, 0.1, 0.2],
                        help="Define ratio in which dataset will be split for training, validation and testing")

        return parser

class Orpheus(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.train_f1 = pl.metrics.F1(args.num_classes, average = 'macro')
        self.val_f1 = pl.metrics.F1(args.num_classes, average = 'macro')
        self.test_f1 = pl.metrics.F1(args.num_classes, average = 'macro')

        self.conf_matrix = np.zeros((args.num_classes, args.num_classes))

    def training_step(self, batch, batch_idx):
        x, target = batch
        x = self(x)

        loss = F.cross_entropy(x, target)
        self.log('train_loss', loss)

        self.train_acc(x, target)
        self.log('train_acc', self.train_acc, on_step = True, on_epoch = False)

        self.train_f1(x, target)
        self.log('train_f1', self.train_f1, on_step = True, on_epoch = False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        x = self(x)

        loss = F.cross_entropy(x, target)
        self.log('val_loss', loss)

        self.val_acc(x, target)
        self.log('val_acc', self.val_acc, on_step = True, on_epoch = True)

        self.val_f1(x, target)
        self.log('val_f1', self.val_f1, on_step = True, on_epoch = True)

        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        x = self(x)

        loss = F.cross_entropy(x, target)
        self.log('test_loss', loss)

        self.test_acc(x, target)
        self.log('test_acc', self.test_acc, on_step = False, on_epoch = True)

        self.test_f1(x, target)
        self.log('test_f1', self.test_f1, on_step = False, on_epoch = True)

        self.conf_matrix += confusion_matrix(torch.argmax(x, dim = -1).cpu().numpy(), target.cpu().numpy(), labels=list(range(self.args.num_classes)))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents = [parent_parser], add_help = False)

        parser.add_argument('--lr', type=float, default = 3e-4,
                        help="Learning rate")
        parser.add_argument('--num_classes', type=int, default = 6,
                        help="Number of possible classes in dataset")

        return parser
