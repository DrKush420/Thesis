import torch
import numpy as np
import random
import utils




def seed_worker(worker_id):
    """Used for enabling deterministic behaviour (see https://pytorch.org/docs/stable/notes/randomness.html)
    """
    worker_seed = torch.initial_seed() % 2**32+worker_id
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

    unlabelled_data = utils.CropsDataset(
        params.unlabelled_images,
        params.input_size,
        params.apply_masks,
        transforms=utils.test_transforms(crop_size=params.input_size,augment=False)
    )
    unlabelled_dataloader = torch.utils.data.DataLoader(
        unlabelled_data,
        batch_size=params.batch_size,
        shuffle=False,#will mess with indices for active learning!!
        pin_memory=True,
        num_workers=params.num_workers,
        worker_init_fn=seed_worker,
    )
    test_data = utils.CropsDataset(
        params.test_images,
        params.input_size,
        params.apply_masks,
        transforms=utils.test_transforms(crop_size=params.input_size,augment=False)
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=params.num_workers,
        worker_init_fn=seed_worker,
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
    params.training_data=training_data
    return train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader

