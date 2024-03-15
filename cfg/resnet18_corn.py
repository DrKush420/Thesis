import utils
import torch
import torch.nn as nn
from torchvision import models


params = utils.Params()

# Network
params.batch_size = 64
params.input_size = (224, 224)
params.num_classes = 2
params.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
params.model.fc = nn.Linear(params.model.fc.in_features, params.num_classes)

# Dataset
params.train_images, params.val_images = utils.get_train_val_split('data/corn/train_val', 0.2, 0)
params.apply_masks = False

# Loss
params.criterion = nn.BCEWithLogitsLoss()

# Optimizer
params.optimizer = torch.optim.Adam(
    params.model.parameters(),
    lr = 1e-5,
)

# Schedule
params.epochs = 30
params.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
    params.optimizer,
    factor = 1.0,   # keep learning rate constant
)
