import utils
import torch
import torch.nn as nn
from torchvision import models


params = utils.Params()

# Network
params.batch_size = 32
params.input_size = (224, 224)
params.num_classes = 2
params.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
params.model.classifier[1] = nn.Linear(params.model.classifier[1].in_features, params.num_classes)

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
params.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([params.weight_dont_spray, params.weight_spray]))

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

