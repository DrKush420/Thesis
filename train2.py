import torch
import numpy as np
import random
from torchvision import  models
import time
import shutil
import os
import copy
import pickle
import argparse
from glob import glob
from datetime import date
import logging
import csv
import utils
from datetime import datetime
import time



def initialize_params():
    params = utils.Params()

    # Network
    params.batch_size = 32
    params.input_size = (224, 224)
    params.num_classes = 2
    params.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    params.model.classifier[1] = torch.nn.Linear(params.model.classifier[1].in_features, params.num_classes)

    # Dataset
    #params.train_images, params.val_images = utils.get_train_val_split('D:/data/wheat/train_val_squares')#, 0.2, 0)
    params.train_images,params.unlabelled_images,params.test_images,params.val_images = utils.get_datasets_split('./data/wheat/train_val_squares')
    params.apply_masks = False
    params.train_transform = utils.train_transforms(crop_size=params.input_size, mean=0.5, std=0.5)
    params.test_transform = utils.test_transforms(crop_size=params.input_size, mean=0.5, std=0.5)

    # Loss
    # pos_weight is calculated based on #dont_spray/#spray samples in the dataset
    params.weight_dont_spray = 1.0
    params.weight_spray = 2.2
    #params.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([params.weight_dont_spray, params.weight_spray]))
    #params.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor( [params.weight_spray]))
    params.criterion = torch.nn.CrossEntropyLoss()#weight=torch.FloatTensor([params.weight_dont_spray, params.weight_spray]))

    # Optimizer
    params.weight_decay = 0.05
    params.optimizer = torch.optim.AdamW(
        params.model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=params.weight_decay
    )

    # Schedule
    params.epochs = 50
    params.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        params.optimizer,
        factor=0.1,
        mode='max',
        patience=10,
        threshold=0.001,
    )
    
    return params

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
        shuffle=True,
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

def uncertainty_training(cycles,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader,params,writer):
    
    for i in range(0,cycles):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed)
        uncertain_ind=utils.select_uncertain(params, unlabelled_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=100)
        train_dataloader.dataset.add_data(unlabelled_dataloader.dataset.get_data(uncertain_ind))
        unlabelled_dataloader.dataset.remove_data(uncertain_ind)
        acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
        writer.writerow([acc,recall, precision, i*100+200, "active learning_no_weight", "uncertainty_monte_carlo",seed])
        params = initialize_params()

def clean_data(cycles,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader
               ,params,file):
    last_accuracy=0
    for i in range(0,cycles):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed)
        acc,_,_=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
        if (last_accuracy-acc)>2:
            for item in uncertain_data:
                file.write(f"{item}\n")
            file.flush()
        last_accuracy=acc
        uncertain_ind=utils.select_uncertain(params, unlabelled_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=100)
        uncertain_data=unlabelled_dataloader.dataset.get_data(uncertain_ind)
        train_dataloader.dataset.add_data(uncertain_data)
        unlabelled_dataloader.dataset.remove_data(uncertain_ind)
        params = initialize_params()

def random_training(cycles,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader,params,writer):
    for i in range(0,cycles):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed)
        indexes=utils.select_random(unlabelled_dataloader.dataset.__len__(),size=100)
        train_dataloader.dataset.add_data(unlabelled_dataloader.dataset.get_data(indexes))
        unlabelled_dataloader.dataset.remove_data(indexes)
        acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
        writer.writerow([acc,recall, precision, i*100+200, "active learning_no_weight", "random",seed])
        params = initialize_params()

def set_deterministic(seed):
    logging.info('Deterministic training enabled')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def embedding(datapath):
    all_images = sorted(glob(os.path.join(datapath, '*/*.jpg')))
    random.shuffle(all_images)
    #all_images=all_images[:15000]
    data = utils.CropsDataset2(
        all_images,
        norm=utils.test_transforms(crop_size=(224, 224)),
        transforms=utils.embedded_transforms(crop_size=(224, 224))
    )
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=5,
        worker_init_fn=seed_worker,
    )
    utils.train_embedding(dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("-c", "--config", help="Path to training config file",default="cfg/mobilenetv2_wheat_squares.py")
    parser.add_argument('-l', '--logdir', help='Logging folder for this experiment', default="logs")
    parser.add_argument('-d', '--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('-m', '--manual_seed', type=int, help='Set manual seed value for random generators', default=0)
    parser.add_argument('-sl', '--seedlist', action='store_true', help='Execute with all seeds in list')
    args = parser.parse_args()
    
    
    
    sleeptime = 8 * 60 * 60
    # Pause the program for 8 hours
    #time.sleep(sleeptime)

    # create logging directory
    tb_dir_name, checkpoints_dir_name = utils.prepare_logdir(args.logdir, args.config)

    # Set deterministic mode if required
    if args.deterministic:
        set_deterministic(args.manual_seed)
    
    # parse config
    params = utils.get_config_from_path(args.config)

    # prepare datasets and dataloaders
    #train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    columns = [ 'test accuracy','recall','precision', 'trainingset_size', 'type', 'method','seed']

    #logfile
    """
    file=open(args.logdir+"/modeldata/"+datetime.now().strftime("_%B%d_%H_%M_%S_")+".csv", 'w')
    writer = csv.writer(file)
    writer.writerow(columns)
    """

    file = open("logs/shitdata.txt", 'a')
    # Start the training
    #active learning uncertainty

    #seedlist=[69,420,1337,42069,69420]
    #seedlist=[0,1,2,3,4,5,6,7,8,9]
    #seedlist=[10,11,12,13,14,15,16,17,18,19]
    seedlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    if args.seedlist:
        print("Running for all seeds in list-->",seedlist)
        for seed in seedlist:
            """
            set_deterministic(seed)
            params=initialize_params()
            train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
            uncertainty_training(10,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader,params,writer)
            file.flush()
            set_deterministic(seed)
            params=initialize_params()
            train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
            random_training(10,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader,params,writer)
            file.flush()
            """
    seed=0
    while(True):
        params=initialize_params()
        train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
        clean_data(10,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader,params,file)
        seed+=1

    file.close()


    
    #embedding('./data/wheat/train_val_squares')




    #uncertainty_training(10)
    #params.train_images,params.unlabelled_images,params.test_images,params.val_images = utils.get_datasets_split('D:/data/wheat/train_val_squares')
    #train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    #random_training(10)

        
