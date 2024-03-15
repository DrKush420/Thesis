import torch
import numpy as np
import random
from torchvision import datasets, models
import time
import shutil
import os
import copy
import pickle
import argparse
from glob import glob
from datetime import date
import logging

import utils


def seed_worker(worker_id):
    """Used for enabling deterministic behaviour (see https://pytorch.org/docs/stable/notes/randomness.html)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(params):
    """Create dataset and dataloader for training and validation
    """
    try:
        train_transform = params.train_transform
    except AttributeError:
        train_transform = utils.train_transforms(crop_size=params.input_size)

    training_data = utils.CropsDataset(
        params.train_images,
        params.input_size,
        params.apply_masks,
        transforms=train_transform
    )

    try:
        test_transform = params.test_transform
    except AttributeError:
        test_transform = utils.test_transforms(crop_size=params.input_size)

    validation_data = utils.CropsDataset(
        params.val_images,
        params.input_size,
        params.apply_masks,
        transforms=test_transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        training_data,
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=params.num_workers,
        worker_init_fn=seed_worker,
    )

    val_dataloader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=params.num_workers,
        worker_init_fn=seed_worker,
    )

    return train_dataloader, val_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("-c", "--config", help="Path to training config file")
    parser.add_argument('-l', '--logdir', help='Logging folder for this experiment', default="logs")
    parser.add_argument('-d', '--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('-m', '--manual_seed', type=int, help='Set manual seed value for random generators', default=0)
    args = parser.parse_args()

    # create logging directory
    tb_dir_name, checkpoints_dir_name = utils.prepare_logdir(args.logdir, args.config)

    # Set deterministic mode if required
    if args.deterministic:
        logging.info('Deterministic training enabled')
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # parse config
    params = utils.get_config_from_path(args.config)

    # prepare datasets and dataloaders
    train_dataloader, val_dataloader = create_dataloaders(params)

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start the training
    utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name)
