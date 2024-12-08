import os
from .lt_data import LT_Dataset
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .randaugment import RandomAugment
from PIL import Image

    
class ImageNet_LT(LT_Dataset):
    classnames_txt = "./datasets/ImageNet_LT/classnames.txt"
    train_txt = "./datasets/ImageNet_LT/ImageNet_LT_train.txt"
    test_txt = "./datasets/ImageNet_LT/ImageNet_LT_test.txt"

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


class ImageNet_Augmentention(Dataset):
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

