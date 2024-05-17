import torch
import numpy as np
import random
import argparse
import logging
import csv
import utils
from datetime import datetime
from init_models import auxiliary , initialize_params
from dataloaders import create_dataloaders,create_unlabelled_dataloader



def disagreement(seed=None,corntest=False):
    params_aux = auxiliary()
    if seed !=None:
        set_deterministic(seed)
    params=initialize_params(startsize,True,root)
    train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    for i in range(steps):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="disagreement_",network="primary_")
        utils.train_classifier(params_aux, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="disagreement_",network="auxiliary_")
        if corntest:
            acc,precision,recall,indices=utils.test_with_tracking(params, test_dataloader, device,checkpoints_dir_name,network="primary_")
            oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06=utils.years(unlabelled_dataloader.dataset.get_alldata())
            moud2021,m2021_05,m2021_06,m2022_05,m2022_06,m2023_06=utils.years(test_dataloader.dataset.get_data(indices))
            writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), "active learning", "disagreement",seed,
                             train_dataloader.dataset.get_fraction(),oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06,
                             moud2021,m2021_05,m2021_06,m2022_05,m2022_06,m2023_06])
        else:
            acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name,network="primary_")
            writer.writerow([acc,recall, precision, i*stepsize+startsize, "active learning", "disagreement",seed,
                             train_dataloader.dataset.get_fraction()])

        if i==steps-1:
            break
        disagree_indices=utils.select_disagreement(params,params_aux, unlabelled_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,size=stepsize)
        """uncertain and diversity on disagreed set"""
        indices=disagree_indices
        """for now just disagreement"""
        selected_data=unlabelled_dataloader.dataset.get_data(indices)
        train_dataloader.dataset.add_data(selected_data)
        unlabelled_dataloader.dataset.remove_data(indices)
        params = initialize_params(startsize,False,root)
        params_aux = auxiliary()
    file.flush()
    

#test
def clean_data(cycles,seed,train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader
               ,params,file):
    last_accuracy=0
    for i in range(0,cycles):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed)
        acc,_,_=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
        if (last_accuracy-acc)>4:
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

def random_training(seed=None,steps=10,corntest=False):
    if seed !=None:
        set_deterministic(seed)
    params=initialize_params(startsize,True,root)
    train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    for i in range(steps):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="random")
        if corntest:
            acc,precision,recall,indices=utils.test_with_tracking(params, test_dataloader, device,checkpoints_dir_name)
            oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06=utils.years(unlabelled_dataloader.dataset.get_alldata())
            moud2021,m2021_05,m2021_06,m2022_05,m2022_06,m2023_06=utils.years(test_dataloader.dataset.get_data(indices))
            writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), "active learning", "random",seed,
                             train_dataloader.dataset.get_fraction(),oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06,
                             moud2021,m2021_05,m2021_06,m2022_05,m2022_06,m2023_06])
        else:
            acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
            writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), "active learning", "random",
                             seed,train_dataloader.dataset.get_fraction()])
        if i==steps-1:
            break
        indexes=utils.select_random(unlabelled_dataloader.dataset.__len__(),size=stepsize)
        train_dataloader.dataset.add_data(unlabelled_dataloader.dataset.get_data(indexes))
        unlabelled_dataloader.dataset.remove_data(indexes)
        params = initialize_params(startsize,False,root)
    file.flush()


def active_learning(function,type,method,seed=None,steps=10,corntest=False):
    if seed !=None:
        set_deterministic(seed)
    params=initialize_params(startsize,True,root)
    train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    for i in range(steps):
        utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method=method)
        if corntest:
            acc,precision,recall,indices=utils.test_with_tracking(params, test_dataloader, device,checkpoints_dir_name)
            oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06=utils.years(unlabelled_dataloader.dataset.get_alldata())
            moud2021,m2021_05,m2021_06,m2022_05,m2022_06,m2023_06=utils.years(test_dataloader.dataset.get_data(indices))
            writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), type, method,seed,
                             train_dataloader.dataset.get_fraction(),oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06,
                             moud2021,m2021_05,m2021_06,m2022_05,m2022_06,m2023_06])
        else:
            acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
            writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), type, method,seed,train_dataloader.dataset.get_fraction()])
        if i==steps-1:
            break
        indexes=function(params, unlabelled_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,train_dataloader,split_size=stepsize)
        train_dataloader.dataset.add_data(unlabelled_dataloader.dataset.get_data(indexes))
        unlabelled_dataloader.dataset.remove_data(indexes)
        params = initialize_params(startsize,False,root)
    file.flush()
    
def set_deterministic(seed):
    logging.info('Deterministic training enabled')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ssl_test(seed=None,method=""):
    
    if seed !=None:
        set_deterministic(seed)
    params=initialize_params(startsize,True,root)
    train_dataloader, val_dataloader,test_dataloader,unlabelled_dataloader = create_dataloaders(params)
    train_dataloader.dataset.ssl=utils.ssl_transforms(crop_size=params.input_size)
    unlabelled_dataloader=create_unlabelled_dataloader(params,utils.get_unlabelled_data("./data/corn/corn/unlabelled"))#get the true unlabelled dataset and dataloader
    utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="Mean_Teacher",training_function=utils.mean_teacher,unlabelled_dataloader=unlabelled_dataloader)
    acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
    writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), "Mean Teacher", "No Querry Method",seed,train_dataloader.dataset.get_fraction()])

    utils.train_classifier(params, train_dataloader, val_dataloader, device,
                           tb_dir_name, checkpoints_dir_name,seed,method="standard")
    acc,precision,recall=utils.test_model(params, test_dataloader, device,checkpoints_dir_name)
    writer.writerow([acc,recall, precision, train_dataloader.dataset.__len__(), "Normal Training", "No Querry Method",seed,train_dataloader.dataset.get_fraction()])




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("-c", "--config", help="Path to training config file",default="cfg/mobilenetv2_wheat_squares.py")
    parser.add_argument('-l', '--logdir', help='Logging folder for this experiment', default="logs")
    parser.add_argument('-d', '--deterministic', action='store_true', help='Enable deterministic training')
    parser.add_argument('-m', '--manual_seed', type=int, help='Set manual seed value for random generators', default=0)
    parser.add_argument('-sl', '--seedlist', action='store_true', help='Execute with all seeds in list')
    parser.add_argument('-ct', '--corntest', action='store_true', help='corn dataset test')
    parser.add_argument("-root", "--dataroot", help="Path to root of data",default="./data")

    args = parser.parse_args()
    root=args.dataroot
    

    # create logging directory
    tb_dir_name, checkpoints_dir_name = utils.prepare_logdir(args.logdir, args.config)

    # Set deterministic mode if required
    if args.deterministic:
        set_deterministic(args.manual_seed)
    

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.corntest:
        columns = [ 'test accuracy','recall','precision', 'trainingset_size', 'type', 'method','seed',"percentage_spray","doud2021",
                   "d2021_05","d2021_06","d2022_05","d2022_06","d2023_06","moud2021","m2021_05","m2021_06","m2022_05"
                   ,"m2022_06","m2023_06"]

    else:
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
    startsize=500
    stepsize=500
    steps=10
    if args.seedlist:
        print("Running for all seeds in list-->",seedlist)
        for seed in seedlist:
            #ssl_test(seed=seed)
            active_learning(utils. div_unc_trainingset_included,"active learning","unc_div_new",seed=seed,steps=steps,corntest=args.corntest)
            #active_learning(utils.select_uncertain_carlo,"active learning","uncertainty_monte_carlo",seed=seed,steps=steps,corntest=args.corntest)
            #random_training(seed=seed,steps=steps,corntest=args.corntest)
            #active_learning(utils.div_unc,"active learning","div_unc_kCent_greedy",seed=seed,steps=steps,corntest=args.corntest)
            #disagreement(seed=seed,corntest=args.corntest)


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

#original corn set size: 12876
#german corn set size: 9696

    #embedding('./data/wheat/train_val_squares')

