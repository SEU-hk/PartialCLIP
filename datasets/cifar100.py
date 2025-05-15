import os
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision import models

from .__init__ import *
from .lt_data import LT_Dataset
from .randaugment import RandomAugment
from augment.randaugment import RandomAugment
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy

from utils.candidate_set_generation import *
from utils.util import generate_instancedependent_candidate_labels

from models.partial_models.wide_resnet import WideResNet
import datasets


class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_factor=None, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

        if train and imb_factor is not None:
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)

        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list


class CIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=None, train=train, transform=transform)


class CIFAR100_IR10(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform)

class CIFAR100_IR20(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.05, train=train, transform=transform)
        
class CIFAR100_IR50(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform)


class CIFAR100_IR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform)
        

class CIFAR100_IR150(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.0067, train=train, transform=transform)     


class CIFAR_Augmentation(Dataset):
    def __init__(self, data, given_label_matrix, true_labels, strong_tranform, weak_tranform):
        """
        Args:
            images: images
            given_label_matrix: PLL candidate labels
            true_labels: GT labels
            con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        """
        self.data = data
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = weak_tranform
        self.strong_transform = strong_tranform

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index
    

def load_cifar100_id(cfg, transform_train, transform_test):
    original_train = dsets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    ori_data, ori_labels = original_train.data, torch.Tensor(original_train.targets).long()

    test_dataset = dsets.CIFAR100(root='./data', train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg.batch_size, \
        shuffle=False, num_workers=4, pin_memory=False
    )
    
    if 0 < cfg.partial_rate < 1:
        partialY_matrix = fps(ori_labels, cfg.partial_rate)
    else:
        ori_data = torch.Tensor(original_train.data)
        model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
        model.load_state_dict(
            torch.load(os.path.expanduser('weights/CIFAR100.pt'))[
                'model_state_dict'])
        ori_data = ori_data.permute(0, 3, 1, 2)
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels, 0.1)
        ori_data = original_train.data

    num_instances = len(original_train)
    classnames = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    num_classes = len(classnames)
    
    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')
    
    if cfg.pre_filter == True:
        print('Pre Filter !')
        partialY_matrix, ori_data, ori_labels = pre_filter(cfg, partialY_matrix, ori_data, ori_labels)
        
    partial_training_dataset = CIFAR100_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=20,
        pin_memory=False,
        drop_last=True
    )
    
    train_test_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=cfg.batch_size, shuffle=False, num_workers=20)
    
    return partial_training_dataloader, train_test_loader, partialY_matrix, test_loader, num_instances, num_classes, classnames

 
def load_cifar100_lt(cfg, transform_train, transform_test, transform_plain):
    root = cfg.root
    
    train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
    train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
    test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

    train_dataset = train_init_dataset
    num_classes = train_dataset.num_classes
    print("num_classes:", num_classes)
    num_instances = len(train_dataset.labels)
    cls_num_list = train_dataset.cls_num_list
    print(max(cls_num_list), min(cls_num_list))
    classnames = train_dataset.classnames

    data, labels = train_dataset.data, torch.Tensor(train_dataset.labels).long()

    partialY = fps(labels, cfg.partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    print('Average candidate num: ', partialY.sum(1).mean())

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    train_test_loader = DataLoader(train_init_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    partialY[torch.arange(partialY.shape[0]), labels] = 1

    if cfg.pre_filter == True:
        partialY, data, labels = pre_filter(cfg, partialY, data, labels)

    # train_givenY = CIFAR100_Partialize(data, partialY.float(), labels.float())

    print("**********")
    train_givenY = CIFAR_Augmentation(data, partialY.float(), labels.float(), transform_train, transform_plain)
    
    print('Average candidate num: ', partialY.sum(1).mean())

    train_loader = DataLoader(dataset=train_givenY, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, train_test_loader, partialY, test_loader, num_instances, num_classes, classnames, cls_num_list
    
    
class CIFAR100_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])


    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.ori_images[index])
        each_image_s = self.strong_transform(self.ori_images[index])
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index

