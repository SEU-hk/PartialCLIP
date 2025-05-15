import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models
from augment.randaugment import RandomAugment
from augment.autoaugment_extra import ImageNetPolicy
from utils.util import generate_instancedependent_candidate_labels
import torch.nn as nn
from datasets.dataset_cub import Cub2011
from torchvision.datasets.folder import pil_loader
from utils.candidate_set_generation import *


def load_cub200(cfg, transform_train, transform_test):
    original_train = Cub2011(root=cfg.root, train=True, transform=transform_train, loader=pil_loader)
    original_full_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=len(original_train),
                                                       shuffle=False, num_workers=20)
    ori_data, ori_labels = next(iter(original_full_loader))
    ori_labels = ori_labels.long()
    
    num_instances = len(original_train)

    classnames = []
    with open('/home/hekuang/LIFT-main/LIFT2/data/classnames/cub200.txt', 'r') as file:
        for line in file.readlines():
            classnames.append(line.strip())
 
    num_classes = len(classnames)
    print(num_classes)

    test_dataset = Cub2011(root=cfg.root, train=False, transform=transform_test, loader=pil_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)

    if 0 < cfg.partial_rate < 1:
        partialY_matrix = fps(ori_labels, cfg.partial_rate)
      
    else:  
        model = models.wide_resnet50_2()
        model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
        model = model.cuda()
        model.load_state_dict(torch.load('weights/cub200.pt')['model_state_dict'])
        partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels, 0.1)

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1

    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')

    partial_training_dataset = CUB200_Partialize(original_train.root, original_train.base_folder, original_train.data,
                                                 partialY_matrix.float(), ori_labels.float())

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    return partial_training_dataloader, original_full_loader, partialY_matrix, test_loader, num_instances, num_classes, classnames


def cpl_cub200(cfg, transform_train, transform_test):
    original_train = Cub2011(root=cfg.root, train=True, transform=transform_train, loader=pil_loader)
    original_full_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=len(original_train),
                                                       shuffle=False, num_workers=20)
    ori_data, ori_labels = next(iter(original_full_loader))
    ori_labels = ori_labels.long()
    
    num_instances = len(original_train)

    classnames = []
    with open('/home/hekuang/LIFT-main/LIFT2/data/classnames/cub200.txt', 'r') as file:
        for line in file.readlines():
            classnames.append(line.strip())
            
    label_to_idx = {}
    for idx, classname in enumerate(classnames):
        label_to_idx[classname] = idx
 
    num_classes = len(classnames)
    print(num_classes)

    test_dataset = Cub2011(root=cfg.root, train=False, transform=transform_test, loader=pil_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)

    train_test_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=cfg.batch_size,
                                                       shuffle=False, num_workers=20)
    
    return original_train, ori_data, ori_labels, train_test_loader, test_loader, num_instances, num_classes, classnames, label_to_idx


class CUB200_Partialize(Dataset):
    def __init__(self, root, base_folder, data, given_partial_label_matrix, true_labels):
        self.root = root
        self.base_folder = base_folder
        self.data = data
        self.loader = pil_loader

        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        self.distill_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ])
        self.weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1
        img = self.loader(path)

        each_image_w = self.weak_transform(img)
        each_image_s = self.strong_transform(img)
        each_image_distill= self.distill_transform(img)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image_w, each_image_s, each_label, each_true_label, index
