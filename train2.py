import torch
import numpy as np
import random
import argparse
import logging
import csv
import utils
from datetime import datetime


from models import auxiliary, initialize_params
from dataloaders import create_dataloaders


def disagreement(cycles,train_dataloader, val_dataloader,test_dataloader,params
                         ,unlabelled_dataloader,writer,stepsize,startsize,seed=None):
    if seed !=None:
        set_deterministic(seed)
    params_aux = auxiliary()
    for i in range(0,cycles):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="disagreement_",network="primary_")
        utils.train_classifier(params_aux, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="disagreement_",network="auxiliary_")
        acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name,network="primary_")
        writer.writerow([acc,recall, precision, i*stepsize+startsize, "active learning", "disagreement",seed,train_dataloader.dataset.get_fraction()])
        if i==steps-1:
            break
        disagree_indices=utils.select_disagreement(params,params_aux, unlabelled_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,size=stepsize)
        """uncertain and diversity on disagreed set"""
        indices=disagree_indices
        """for now just disagreement"""
        selected_data=unlabelled_dataloader.dataset.get_data(indices)
        train_dataloader.dataset.add_data(selected_data)
        unlabelled_dataloader.dataset.remove_data(selected_data)
        params = initialize_params()
        params_aux = auxiliary()
    


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
                     tb_dir_name, checkpoints_dir_name,split_size=stepsize)
        uncertain_data=unlabelled_dataloader.dataset.get_data(uncertain_ind)
        train_dataloader.dataset.add_data(uncertain_data)
        unlabelled_dataloader.dataset.remove_data(uncertain_ind)
        params = initialize_params()

def random_training(seed=None,steps=10):
    if seed !=None:
        set_deterministic(seed)
    params=initialize_params(startsize,True)
    train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    for i in range(steps):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="random")
        acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
        writer.writerow([acc,recall, precision, i*stepsize+startsize, "active learning", "random",seed,train_dataloader.dataset.get_fraction()])
        if i==steps-1:
            break
        indexes=utils.select_random(unlabelled_dataloader.dataset.__len__(),size=stepsize)
        train_dataloader.dataset.add_data(unlabelled_dataloader.dataset.get_data(indexes))
        unlabelled_dataloader.dataset.remove_data(indexes)
        params = initialize_params(startsize,False)
    file.flush()


def active_learning(function,type,method,seed=None,steps=10):
    if seed !=None:
        set_deterministic(seed)
    params=initialize_params(startsize,True)
    train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    for i in range(0,steps):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method=method)
        acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
        writer.writerow([acc,recall, precision, i*stepsize+startsize, type, method,seed,train_dataloader.dataset.get_fraction()])
        if i==steps-1:
            break
        indexes=function(params, unlabelled_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=stepsize)
        train_dataloader.dataset.add_data(unlabelled_dataloader.dataset.get_data(indexes))
        unlabelled_dataloader.dataset.remove_data(indexes)
        params = initialize_params(startsize,False)
    file.flush()
    
def set_deterministic(seed):
    logging.info('Deterministic training enabled')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# unfinished test
"""       
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
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("-c", "--config", help="Path to training config file",default="cfg/mobilenetv2_wheat_squares.py")
    parser.add_argument('-l', '--logdir', help='Logging folder for this experiment', default="logs")
    parser.add_argument('-d', '--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('-m', '--manual_seed', type=int, help='Set manual seed value for random generators', default=0)
    parser.add_argument('-sl', '--seedlist', action='store_true', help='Execute with all seeds in list')
    args = parser.parse_args()
    
    

    # create logging directory
    tb_dir_name, checkpoints_dir_name = utils.prepare_logdir(args.logdir, args.config)

    # Set deterministic mode if required
    if args.deterministic:
        set_deterministic(args.manual_seed)
    

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    columns = [ 'test accuracy','recall','precision', 'trainingset_size', 'type', 'method','seed',"percentage_spray"]

    #logfile
    
    file=open(args.logdir+"/modeldata/"+datetime.now().strftime("_%B%d_%H_%M_%S_")+".csv", 'w')
    writer = csv.writer(file)
    writer.writerow(columns)
    file.flush()

    #file = open("logs/shitdata.txt", 'a')
    # Start the training
    #active learning uncertainty

    seedlist=[0,1,2,3,4,5,6,7,8,9]
    #seedlist=[10,11,12,13,14,15,16,17,18,19]
    #seedlist=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    startsize=1200
    stepsize=1200
    steps=10
    if args.seedlist:
        print("Running for all seeds in list-->",seedlist)
        for seed in seedlist:
            active_learning(utils.select_uncertain,"active learning","uncertainty",seed=seed,steps=steps)
            active_learning(utils.select_uncertain_carlo,"active learning","uncertainty_monte_carlo",seed=seed,steps=steps)
            random_training(seed=seed,steps=steps)
            #active_learning(seed,utils.DPP_div_unc,"active learning","DPP_diversity_uncertainty",steps=steps)


    """    
    seed=0
    while(True):
        params=initialize_params()
        train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
        clean_data(10,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader
                   ,params,file)
        seed+=1

    file.close()
    """

    
    #embedding('./data/wheat/train_val_squares')

