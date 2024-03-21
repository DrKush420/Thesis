import os
import time
import torch
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import  models
import numpy as np
from torchvision.transforms import v2
from sklearn.metrics import precision_score, recall_score
from dppy.finite_dpps import FiniteDPP
import torch.nn.functional as F


from numpy.random import RandomState
class ClassifierAccuracy():

    def __init__(self):
        self.total = 0
        self.correct = 0

    def __call__(self, outputs, targets):
        _, predicted = outputs.max(1)
        _, labels = targets.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(labels).sum().item()
        acc = self.correct / self.total
        return acc
    

def validation_classifier(model, dataloader, criterion, device):
    """Validate the classifier
    """
    model.eval()
    validation_loss = 0
    calc_accuracy = ClassifierAccuracy()
    progress_bar = tqdm(dataloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            targets=F.one_hot(targets).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
            acc = calc_accuracy(outputs, targets)
            progress_bar.set_description('Val Loss: %.3f | Val Acc: %.3f%%'% (validation_loss / (batch_idx + 1), 100. * acc))

    return (validation_loss / (batch_idx + 1)), acc


def train_epoch_classifier(model, dataloader, optimizer, criterion, device, log_inputs_fn):
    """Train one epoch
    """
    # TODO: add support for inception
    model.train()
    train_loss = 0
    calc_accuracy = ClassifierAccuracy()
    progress_bar = tqdm(dataloader)
    mixup = v2.MixUp(num_classes=2)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        if log_inputs_fn is not None:
            log_inputs_fn(inputs)
            log_inputs_fn = None    # log only once every epoch
        inputs, targets = mixup(inputs, targets)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        acc = calc_accuracy(outputs, targets)
        progress_bar.set_description('Train Loss: %.3f | Train Acc: %.3f%%'% (train_loss / (batch_idx + 1), 100. * acc))

    return (train_loss / (batch_idx + 1)), acc


def train_classifier(params, train_dataloader, val_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,seed=0,method=""):
    """Train a classifier model for a number of epochs on the given device
    """

    tb = SummaryWriter(tb_dir_name)
    start_time = time.time()
    logging.info("Start training")
    best_acc = 0.0

    # ensure model and criterion are on the right device
    params.model = params.model.to(device)
    params.criterion = params.criterion.to(device)

    for epoch in range(params.epochs):
        lr = params.optimizer.param_groups[0]['lr']
        logging.info('Epoch: %d, lr: %f' % (epoch + 1, lr))
        logging.info('Seed: %d' , seed)
        logging.info('method:'+method)
        tb.add_scalar('Train/LearningRate', lr, epoch)

        def log_fn(inputs):
            tb.add_images('preprocessed images', inputs[:5, ...], epoch)

        train_loss, train_acc = train_epoch_classifier(params.model, train_dataloader, params.optimizer, params.criterion, device, log_fn)
        logging.info('Train Loss: %.3f | Train Acc: %.3f%%', train_loss, train_acc * 100)
        tb.add_scalar('Train/Loss', train_loss, epoch)
        tb.add_scalar('Train/Acc', train_acc * 100, epoch)

        val_loss, val_acc = validation_classifier(params.model, val_dataloader, params.criterion, device)
        logging.info('Val Loss: %.3f | Val Acc: %.3f%%', val_loss, val_acc * 100)
        tb.add_scalar('Val/Loss', val_loss, epoch)
        tb.add_scalar('Val/Acc', val_acc * 100, epoch)

        if isinstance(params.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            params.lr_scheduler.step(val_acc)
        else:
            params.lr_scheduler.step()

        if val_acc > best_acc:
            logging.info('Saving..')
            state = {
                'net': params.model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(checkpoints_dir_name, 'best.pt'))
            best_acc = val_acc

    time_elapsed = time.time() - start_time
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return val_acc

def enable_dropout(model):#enable dropout for monte carlo dropout
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def calculate_uncertainty(inputs,model):
    outputs = model(inputs)
    return torch.abs(torch.softmax(outputs, dim=1)[:,1] - 0.5) 


def monte_carlo_uncertainty(inputs,model):

    n_samples=25
    batch_size, *_ = inputs.shape
    predictions = torch.zeros((n_samples, batch_size))

    for i in range(n_samples):
        predictions[i] =(torch.softmax( model(inputs), dim=1))[:,1]
    
    uncertainty = predictions.std(0)
    return  uncertainty



def select_uncertain(params, unl_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=500):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    params.model = params.model.to(device)
    params.model.eval()
    uncertainties = []
    all_indices = [j for j in range(0, unl_dataloader.dataset.__len__())]
    progress_bar = tqdm(unl_dataloader)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            uncertainty = calculate_uncertainty(inputs,params.model)
            uncertainties.extend(uncertainty.cpu().numpy())

    # Sort samples by uncertainty
    indices_uncertainties = list(zip(all_indices, uncertainties))
    indices_uncertainties.sort(key=lambda x: x[1])  

    
    # Split the dataset
    uncertain_indices = [idx for idx, _ in indices_uncertainties[:split_size]]


    
    return uncertain_indices

def select_uncertain_carlo(params, unl_dataloader, device,
                     tb_dir_name, checkpoints_dir_name,split_size=500):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    params.model = params.model.to(device)
    params.model.eval()
    uncertainties = []
    all_indices = [j for j in range(0, unl_dataloader.dataset.__len__())]
    progress_bar = tqdm(unl_dataloader)
    enable_dropout(params.model)#for monte carlo dropout
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            uncertainty=monte_carlo_uncertainty(inputs,params.model)
            uncertainties.extend(uncertainty.cpu().numpy())

    # Sort samples by uncertainty
    indices_uncertainties = list(zip(all_indices, uncertainties))
    indices_uncertainties.sort(key=lambda x: x[1],reverse=True)
    
    # Split the dataset
    uncertain_indices = [idx for idx, _ in indices_uncertainties[:split_size]]

    return uncertain_indices

def test_model(params, test_dataloader, device,checkpoints_dir_name):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    model=params.model.eval()
    model=model.to(device)
    calc_accuracy = ClassifierAccuracy()
    progress_bar = tqdm(test_dataloader)
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            all_targets.extend(targets.numpy())
            inputs, targets = inputs.to(device), targets.to(device)
            targets=F.one_hot(targets).float()
            outputs = model(inputs)
            predicted_classes = outputs.argmax(dim=1)
            all_outputs.extend(predicted_classes.cpu().numpy())          
            acc = calc_accuracy(outputs, targets)
            progress_bar.set_description(f'Test Acc: {100. * acc:.3f}%')
    precision = precision_score(all_targets, all_outputs, pos_label=1)
    recall = recall_score(all_targets, all_outputs, pos_label=1)
    return acc*100,precision,recall

def select_random(len,size=500):
    indexes = np.random.choice(range(len), size=size, replace=False)
    return indexes



def train_embedding(dataloader):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    criterion = torch.nn.TripletMarginLoss(margin=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloader)
        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):

            optimizer.zero_grad()
        
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
            positive_embedding = F.normalize(positive_embedding, p=2, dim=1)
            negative_embedding = F.normalize(negative_embedding, p=2, dim=1)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f'Train Loss: {running_loss / (batch_idx + 1):.3f}')
        filename = f'logs/embeddings/epoch_{epoch}.pt'
        torch.save(model.state_dict(),filename)

def get_vectors(params, dataloader, device,checkpoints_dir_name):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    model=params.model.eval()
    model.classifier = torch.nn.Identity()
    progress_bar = tqdm(dataloader)
    progress_bar.set_description("get vectors")
    model.to(device)
    vectors=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            outputs = model(inputs)
            vectors.extend(outputs)

        
    return vectors

def get_uncertainty(params, unl_dataloader, device,
                     checkpoints_dir_name):
    state = torch.load(os.path.join(checkpoints_dir_name, 'best.pt'))
    params.model.load_state_dict(state['net'])
    params.model = params.model.to(device)
    params.model.eval()
    uncertainties = []
    progress_bar = tqdm(unl_dataloader)
    progress_bar.set_description("get uncertainty")
    #enable_dropout(params.model)#for monte carlo dropout
    with torch.no_grad():
        for i, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            #uncertainty=monte_carlo_uncertainty(inputs,params.model)
            uncertainty = calculate_uncertainty(inputs,params.model)
            uncertainties.extend(uncertainty)

    
    return uncertainties

def DPP_div_unc(params, unl_dataloader, device,
                     checkpoints_dir_name,size=500):
    uncertainties=get_uncertainty(params, unl_dataloader, device,checkpoints_dir_name)
    vectors=get_vectors(params, unl_dataloader, device,checkpoints_dir_name)
    num_items=unl_dataloader.dataset.__len__()
    L = np.zeros((num_items,num_items ))
    logging.info("calculate matrix")
    """
    for i in range(num_items):
        for j in range(num_items):
            dot_product = np.dot(vectors[i], vectors[j])
            norm_i = np.linalg.norm(vectors[i])
            norm_j = np.linalg.norm(vectors[j])
            similarity = dot_product / (norm_i * norm_j)
            uncertainty=max(uncertainties[i],uncertainties[j])
            L[i, j] = 0.4*uncertainty + 0.6*similarity
    """
    vectors = torch.stack(vectors).to(device)
    uncertainties = torch.stack(uncertainties).to(device)
    # Normalize vectors to unit norm
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    normalized_vectors = vectors / norms

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t()).cpu().numpy()

    #Expand uncertainties to a matrix
    uncertainty_matrix = torch.max(uncertainties.unsqueeze(1), uncertainties.unsqueeze(0)).cpu().numpy()

    # Combine similarity and uncertainty
    L = 0.4 * uncertainty_matrix + 0.6 * similarity_matrix
    logging.info("DPP")
    dpp = FiniteDPP('likelihood', L=L)
    rng = RandomState(123)
    # Sample x diverse items    75.5
    dpp.sample_mcmc_k_dpp(size=size, random_state=rng)  
    #dpp.sample_exact_k_dpp(size=size)
    sampled_indices = dpp.list_of_samples[-1]
    return sampled_indices
