import os
from .lt_data import LT_Dataset
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.candidate_set_generation import *
from torchvision import transforms
from .randaugment import RandomAugment
from PIL import Image
from .__init__ import *

class Places_LT(LT_Dataset):
    classnames_txt = "./datasets/Places_LT/classnames.txt"
    train_txt = "./datasets/Places_LT/Places_LT_train.txt"
    test_txt = "./datasets/Places_LT/Places_LT_test.txt"

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train, transform)

        self.classnames = self.read_classnames()

        self.names = []
        self.labels = []
        self.data = []
        with open(self.txt) as f:
            for line in f:
                path, label = line.split()
                self.names.append(self.classnames[int(label)])
                self.labels.append(int(label))

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        return image, label, name

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames


class Places_Augmentention(Dataset):
    def __init__(self, image_paths, given_label_matrix, true_labels, strong_tranform, weak_tranform):
        """
        Args:
            images: images
            given_label_matrix: PLL candidate labels
            true_labels: GT labels
            con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        """
        self.image_paths = image_paths
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = weak_tranform
        self.strong_transform = strong_tranform

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index
    
    
    
def load_places_lt(cfg, transform_train, transform_test, transform_plain):
    root = cfg.root
    
    train_dataset = Places_LT(root, train=True, transform=transform_train)
    train_init_dataset = Places_LT(root, train=True, transform=transform_plain)
    test_dataset = Places_LT(root, train=False, transform=transform_test)

    train_dataset = train_init_dataset
    num_classes = train_dataset.num_classes
    print("num_classes:", num_classes)
    num_instances = len(train_dataset.labels)
    cls_num_list = train_dataset.cls_num_list
    print(max(cls_num_list), min(cls_num_list))
    classnames = train_dataset.classnames

    data, labels = train_dataset.img_path, torch.Tensor(train_dataset.labels).long()

    if 0 < cfg.partial_rate < 1:
        partialY = fps(labels, cfg.partial_rate)
    elif cfg.partial_rate == 0:
        partialY = uss(labels)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    print('Average candidate num: ', partialY.sum(1).mean())

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    train_test_loader = DataLoader(train_init_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    partialY[torch.arange(partialY.shape[0]), labels] = 1

    if cfg.pre_filter == True:
        partialY, data, labels = pre_filter(cfg, partialY, data, labels)

    train_givenY = Places_Augmentention(data, partialY.float(), labels.float(), transform_train, transform_plain)

    print('Average candidate num: ', partialY.sum(1).mean())

    train_loader = DataLoader(dataset=train_givenY, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    return train_loader, train_test_loader, partialY, test_loader, num_instances, num_classes, classnames, cls_num_list