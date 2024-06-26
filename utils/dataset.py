import torch.nn.functional as F
import torch
import logging
import glob
import os
import cv2
import numpy as np
import imagesize
import random
from pathlib import Path
import pdb
from torchvision import transforms as t


def get_unlabelled_data(path):
    data=sorted(glob.glob(os.path.join(path, '**/*.jpg'),recursive=True))
    print(len(data))
    return data

def cornssl(root,seed=0,pop=0):
    all_images=sorted(glob.glob(os.path.join(root+'/corn/corn/german_data/2023_06', '*/*.jpg')))
    print(len(all_images))
    #np.random.seed(seed)
    gen = np.random.default_rng(seed)                                  # new random generator with known seed
    gen.shuffle(all_images)   
    groups = np.array_split(all_images, 5)    # 20% of total dataset is testset
    test_images = groups.pop(pop)  
    images = np.concatenate(groups)   
    groups = np.array_split(images, 10) 
    validation_images = groups.pop(pop)  
    training_images = np.concatenate(groups) 
    
    unlabelled_images = []
    return training_images, unlabelled_images, test_images,validation_images

def get_corntest(root,seed=1,test_group=0):
    corn=root+'/corn/corn'
    original_cornset=sorted(glob.glob(os.path.join(corn+'/train_val', '**/*.jpg'),recursive=True))
    print(f'original corn set size: {len(original_cornset)}')
    german_corn=sorted(glob.glob(os.path.join(corn+'/german_data', '**/*.jpg'),recursive=True))
    print(f'german corn set size: {len(german_corn)}')
    gen = np.random.default_rng(seed)                                  # new random generator with known seed
    gen.shuffle(original_cornset)   
    gen.shuffle(german_corn)
    groups = np.array_split(german_corn, 5)    # 20% of dataset is testset
    test_images = groups.pop(test_group)  
    images = np.concatenate(groups)   
    training_images=original_cornset[:5000]
    validation_images = np.concatenate([original_cornset[5000:6000], images[:100]])
    unlabelled_images = np.concatenate([original_cornset[6000:], images[100:]])
    print(len(unlabelled_images))
    print(len(training_images))


    return training_images, unlabelled_images, test_images,validation_images

def get_train_val_split(root, val_fraction=0.2, val_group=0):
    """Return two lists of images: train_images, val_images
    root:           Path of the dataset
    val_fraction:   Fraction of validation images (value between 0.0 and 1.0)
    val_group:      The dataset is split in `1/val_fraction` groups. This number
                    selects the given group of images to be the validation set.
                    The remaining groups are combined into the training set.
                    Changing the validation set is usefull for cross validation.
    NOTE that the splits are made using random choice with constant and known manual seed
    to get deterministic grouping results each time the function is called.
    """
    all_images = sorted(glob.glob(os.path.join(root, '*/*.jpg')))   # sort them to ensure determinism
    gen = np.random.default_rng(1)                                  # new random generator with known seed
    gen.shuffle(all_images)                                         # shuffle images
    groups = np.array_split(all_images, round(1/val_fraction))      # create 1/val_fraction groups of images
    val_images = groups.pop(val_group)                              # select one group for the validation set
    train_images = np.concatenate(groups)                           # use the remaining groups for training set
    return train_images, val_images



def get_datasets_split(root, training_size=500,validation_size=1000,seed=1,test_group=0,more_data=True):
    extra_data_path=root+'/wheat/extra_wheat'
    all_images = sorted(glob.glob(os.path.join(root+'/wheat/train_val_squares', '*/*.jpg')))
    if more_data:
        extra=sorted(glob.glob(os.path.join(extra_data_path, '**/*.jpg'),recursive=True))
        print(len(extra))
        all_images.extend(extra)
    print(len(all_images))
    #np.random.seed(seed)
    gen = np.random.default_rng(seed)                                  # new random generator with known seed
    gen.shuffle(all_images)   
    groups = np.array_split(all_images, 5)    # 20% of total dataset is testset
    test_images = groups.pop(0)  
    images = np.concatenate(groups)   
    training_images = images[:training_size]
    validation_images = images[training_size:training_size+validation_size]
    unlabelled_images = images[training_size+validation_size:]
    

    return training_images, unlabelled_images, test_images,validation_images

def get_mask(mask_coords, mask_shape):
    """Get binary trapezoid mask based on mask_coords and `mask_shape`
    mask_coords:    Tuple of 4 x-values: x_top_left, x_bottom_left, x_bottom_right, x_top_right
    mask_shape:     Tuple: (h, w)
    """
    h, w = mask_shape
    mask = np.zeros((h, w))
    pts = np.array([
        [mask_coords[0], 0],    # top left
        [mask_coords[1], h],    # bottom left
        [mask_coords[2], h],    # bottom right
        [mask_coords[3], 0],    # top right
    ])
    return cv2.drawContours(mask, [pts], 0, (255,), -1)


def get_optimal_network_input_shape(img_shape, area=224*224, down_scale_fact=32):
    """
    img_shape: (h, w) shape of the image to figure out the aspect ratio
    area:      number of square pixels the network input should contain
    Returns:   (net_h, net_w), where net_h*net_w equals `area` (approximately), with an aspect ratio that closely matches the
               aspect ratio of `img_shape` and where h and w are multiples of `down_scale_fact`
    """
    h, w = img_shape
    aspect = w / h
    h_net = np.sqrt(area / aspect)
    w_net = aspect * h_net
    shape = np.array([h_net, w_net])

    return tuple((down_scale_fact * np.round(shape / down_scale_fact)).astype('int32'))

def years(data):

    oud2021=0
    d2021_05=0
    d2021_06=0
    d2022_05=0
    d2022_06=0
    d2023_06=0
    for filename,_,_ in data:

        parts=filename.split('/')
        if parts[3]=="train_val":
            oud2021+=1
        elif parts[4]=="2021_05":
            d2021_05+=1
        elif parts[4]=="2021_06":
            d2021_06+=1
        elif parts[4]=="2022_05":
            d2022_05+=1
        elif parts[4]=="2022_06":
            d2022_06+=1
        elif parts[4]=="2023_06":
            d2023_06+=1
    if (oud2021+d2021_05+d2021_06+d2022_05+d2022_06+d2023_06)!= len(data):
        print('fout ')
    return oud2021,d2021_05,d2021_06,d2022_05,d2022_06,d2023_06

class CropsDataset(torch.utils.data.Dataset):

    def __init__(self, filenames, model_input_shape, apply_masks, transforms=None):
        """
        filenames (list):   List of image file names this dataset should contain
                            Expecting the parent folder of the images to be either 'spray' or 'dont'.
                            Labels are extracted based these parent folder names.
        model_input_shape:  h, w input shape of the model
        apply_masks:        If masks should be applied or not
        transforms:         Image transforms, to be applied on a ROI
        """
        super().__init__()
        self.transforms = transforms
        self.apply_masks = apply_masks
        self.ssl=None #give tranforms right before ssl methods
        self.tensor=t.ToTensor()
        # Build a list with tuples, where each tuple = (filename, mask_id, label)
        self.data = []
        mask_info = {}
        for filename in filenames:
            # get labels and mask coords from filename, example: dont/Wheat_20210903141658_4_4452_25_312_0_415.jpg
            filename = Path(filename)
            label = 1 if filename.parent.stem == 'spray' else 0
            basename = filename.stem
            mask_coords = None
            if apply_masks:
                mask_coords = tuple([int(c) for c in basename.split('_')[-4:]])
                mask_info[mask_coords] = str(filename)
            self.data.append((str(filename), mask_coords, label))

        # Pre-calculate the masks at model_input_shape resolution and find crop shapes
        self.masks = {}
        crop_shapes = set()
        net_h, net_w = model_input_shape
        for mask_coords, filename in mask_info.items():
            w, h = imagesize.get(filename)
            crop_shapes.add((h, w))
            mask = get_mask(mask_coords, (h, w))
            self.masks[mask_coords] = cv2.resize(mask, (net_w, net_h))
    
    def add_data(self,newdata):
        self.data.extend(newdata)

    def get_data(self,indexes):
        return [self.data[i] for i in indexes]
    def get_fraction(self):#get don't/spray fraction of dataset
        labels = np.array([item[2] for item in self.data])
        pos_label_fraction = np.mean(labels == 1)
        return pos_label_fraction
    
    def get_alldata(self):
        return self.data

    def remove_data(self,indexes):
        indexes = set(indexes) #for increased speed 
        self.data = [item for idx, item in enumerate(self.data) if idx not in indexes]



    
    def __getitem__(self, idx):
        filename, mask_coords, label = self.data[idx]

        img = cv2.imread(filename)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # apply ROI mask
        if self.apply_masks:
            mask = self.masks[mask_coords]
            img[mask == 0] = 0

        img = img.transpose((2, 0, 1))  # channel first

        label = torch.as_tensor(label).long()
        #target = F.one_hot(label, num_classes=2).float()
        if self.ssl:
            img2 = cv2.imread(filename)
            img2=self.ssl(image=img2)['image']
            img2 = img2.transpose((2, 0, 1))
            img=np.stack([self.tensor(img),self.tensor(img2)])

        return img, label

    def __len__(self):
        return len(self.data)
    
#word (nog)niet gebruikt
class CropsDataset2(torch.utils.data.Dataset):

    def __init__(self, data,norm,  transforms=None):

        self.data =  [(data[i], data[j]) for i in range(len(data)) for j in range(i+1, len(data))]
        self.transforms = transforms
        self.norm=norm
        print(len(data))
        print(len(self.data))
        #random.shuffle(self.data)
        self.data = random.sample(self.data, int(len(self.data)/10))
        print(len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = cv2.imread(self.data[idx][0])
        neg=cv2.imread(self.data[idx][1])
        anchor=self.norm(image=anchor)['image']
        pos = self.transforms(image=anchor)['image']
        neg=self.norm(image=neg)['image']
        anchor = anchor.transpose((2, 0, 1))  # channel first
        pos = pos.transpose((2, 0, 1))
        neg=neg.transpose((2, 0, 1))

        return anchor,pos ,neg

class CropsUnlabelledDataset(torch.utils.data.Dataset):


    def __init__(self, filenames, transforms=None,transforms2=None):
        super().__init__()
        self.transforms = transforms
        self.tensor=t.ToTensor()
        self.transforms2=transforms2

        # Build a list with tuples, where each tuple = (filename, mask_id, label)
        self.data = []

        for filename in filenames:
            filename = Path(filename)


            self.data.append(str(filename))



    
    def __getitem__(self, idx):
        filename = self.data[idx]

        img = cv2.imread(filename)

        if self.transforms:
            img = self.transforms(image=img)['image']


        img = img.transpose((2, 0, 1))  # channel first
        
        img2 = cv2.imread(filename)
        img2=self.transforms2(image=img2)['image']
        img2 = img2.transpose((2, 0, 1))
        img=np.stack([self.tensor(img),self.tensor(img2)])#np stack
        print(f'Test Acc: {img.shape}')
        return img

    def __len__(self):
        return len(self.data)