import sys
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
import pickle
import numpy as np
from scipy.special import comb
from .randaugment import RandomAugment
from .cutout import Cutout
from scipy.io import loadmat


ImageFile.LOAD_TRUNCATED_IMAGES = True

class PLCIFAR10_Aggregate(Dataset):
    def __init__(self, root, args=None):


        dataset_path = os.path.join(root, 'plcifar10', f"plcifar10.pkl")
        partial_label_all = pickle.load(open(dataset_path, "rb"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))
        ])

        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, None),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))
        ])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        self.input_dim = 32 * 32 * 3
        self.num_classes = 10

        original_dataset = dsets.CIFAR10(root=root, train=True, download=True)
        self.data = original_dataset.data

        self.partial_targets = np.zeros((len(self.data), self.num_classes))

        for key, value in partial_label_all.items():
            for candidate_label_set in value:
                for label in candidate_label_set:
                    self.partial_targets[key, label] = 1


        self.ord_labels = original_dataset.targets
        self.ord_labels = np.array(self.ord_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        original_image = self.test_transform(image)
        weak_image = self.transform(image)
        strong_image=self.strong_transform(image)
        return weak_image, strong_image, self.partial_targets[index], self.ord_labels[index]


class PLCIFAR10_Vaguest(Dataset):
    def __init__(self, root, args=None):

        dataset_path = os.path.join(root, 'plcifar10', f"plcifar10.pkl")

        partial_label_all = pickle.load(open(dataset_path, "rb"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))
        ])

        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, None),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))
        ])

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4922, 0.4832, 0.4486), (0.2456, 0.2419, 0.2605))])
        self.input_dim = 32 * 32 * 3
        self.num_classes = 10

        original_dataset = dsets.CIFAR10(root=root, train=True, download=True)
        self.data = original_dataset.data

        self.partial_targets = np.zeros((len(self.data), self.num_classes))


        for key, value in partial_label_all.items():
            vaguest_candidate_label_set = []
            largest_num = 0
            for candidate_label_set in value:
                if len(candidate_label_set) > largest_num:
                    vaguest_candidate_label_set = candidate_label_set
                    largest_num = len(candidate_label_set)
            for label in vaguest_candidate_label_set:
                self.partial_targets[key, label] = 1            

        self.ord_labels = original_dataset.targets
        self.ord_labels = np.array(self.ord_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        original_image = self.test_transform(image)
        weak_image = self.transform(image)
        strong_image=self.strong_transform(image)
        return weak_image, strong_image, self.partial_targets[index], self.ord_labels[index]
    
