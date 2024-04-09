import os
import time
import torch
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F

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
                     tb_dir_name, checkpoints_dir_name,seed=0,method="",network="",training_function=train_epoch_classifier,unlabelled_dataloader=None):
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
        logging.info('method:'+method+network+training_function.__name__)
        tb.add_scalar('Train/LearningRate', lr, epoch)

        def log_fn(inputs):
            tb.add_images('preprocessed images', inputs[:5, ...], epoch)
        if (unlabelled_dataloader!=None):
            train_loss, train_acc = training_function(params.model, train_dataloader, params.optimizer, params.criterion, device, unlabelled_dataloader)
        else:
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
            torch.save(state, os.path.join(checkpoints_dir_name,network+'best.pt'))
            best_acc = val_acc

    time_elapsed = time.time() - start_time
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return val_acc


def test_model(params, test_dataloader, device,checkpoints_dir_name,network=''):
    state = torch.load(os.path.join(checkpoints_dir_name,network+'best.pt'))
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

