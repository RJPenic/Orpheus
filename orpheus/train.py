import torch

import pytorch_lightning as pl
from orpheus import Orpheus, TrainingDataModule
from han_model import HAN_Model

import argparse
import numpy as np

def add_run_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents = [parent_parser], add_help = False)

    parser.add_argument('--save_path', type=str, default = None,
                    help="Save model parameters in this file")

    return parser

def create_arguments():
    parser = argparse.ArgumentParser()

    parser = HAN_Model.add_model_specific_args(parser)
    parser = Orpheus.add_model_specific_args(parser)
    parser = TrainingDataModule.add_data_module_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_run_specific_args(parser)

    return parser.parse_args()

def main(args):
    trainer = pl.Trainer.from_argparse_args(args)

    data = TrainingDataModule(args)

    model = HAN_Model(data.dataset.get_subset_vocab(data.train_data.indices), args)

    trainer.fit(model, datamodule=data)

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path) # save model

    trainer.test()
    print("-- Confusion matrix --")
    print(model.conf_matrix.astype(int))


if __name__ == "__main__":
    args = create_arguments()

    main(args)
