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
from scipy.io import loadmat

TABULAR_DATASETS = [
    "FGNET",
    "Lost",
    "MSRCv2",
    "Mirflickr",
    "Birdsong",
    "Malagasy",
    "SoccerPlayer",
    "Italian",
    "YahooNews",
    "English"
]

class Tabular_Dataset(Dataset):
    def __init__(self, root, args=None):
        self.dataset_path = root
        self.total_data = loadmat(self.dataset_path)

        self.data, self.ord_labels_mat, self.partial_targets = self.total_data['data'], self.total_data['target'], self.total_data['partial_target']
        self.ord_labels_mat = self.ord_labels_mat.transpose()
        self.partial_targets = self.partial_targets.transpose()
        if self.data.shape[0] != self.ord_labels_mat.shape[0] or self.data.shape[0] != self.partial_targets.shape[0]:
            raise RuntimeError('The shape of data and labels does not match!')
        if self.ord_labels_mat.sum() != len(self.data):
            raise RuntimeError('Data may have more than one label!')
        _, self.ord_labels = np.where(self.ord_labels_mat == 1)
        self.data = (self.data - self.data.mean(axis=0, keepdims=True))/self.data.std(axis=0, keepdims=True)
        self.input_dim = self.data.shape[1]
        self.num_classes = self.ord_labels_mat.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        original_image = image
        weak_image = image
        strong_image = image
        return original_image, weak_image, strong_image, self.partial_targets[index], self.ord_labels[index]
    
    
class gen_index_train_tabular_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.data = images
        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.partial_targets = given_label_matrix
        self.ord_labels = true_labels
        self.input_dim = images.shape[1]
        self.num_classes = given_label_matrix.shape[1]
        
    def __len__(self):
        return len(self.ord_labels)
        
    def __getitem__(self, index):
        each_image = self.data[index]
        each_label = self.partial_targets[index]
        each_true_label = self.ord_labels[index]
        
        return each_image, each_image, each_label, each_true_label, index
    
    
class gen_index_test_tabular_dataset(Dataset):
    def __init__(self, images, true_labels):
        self.data = images
        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.ord_labels = true_labels
        
    def __len__(self):
        return len(self.ord_labels)
        
    def __getitem__(self, index):
        each_image = self.data[index]
        each_true_label = self.ord_labels[index]
        
        return each_image, each_true_label
    
    
def tabular_train_test_dataset_gen(root, seed, cfg):
    dataset_path = os.path.join(root, (cfg.dataset+".mat"))
    total_data = loadmat(root)
    data, ord_labels_mat, partial_targets = total_data['data'], total_data['target'], total_data['partial_target']
    ord_labels_mat = ord_labels_mat.transpose()
    partial_targets = partial_targets.transpose()
    if type(ord_labels_mat) != np.ndarray:
        ord_labels_mat = ord_labels_mat.toarray()
        partial_targets = partial_targets.toarray()
    if data.shape[0] != ord_labels_mat.shape[0] or data.shape[0] != partial_targets.shape[0]:
        raise RuntimeError('The shape of data and labels does not match!')
    if ord_labels_mat.sum() != len(data):
        raise RuntimeError('Data may have more than one label!')
    _, ord_labels = np.where(ord_labels_mat == 1)
    data = (data - data.mean(axis=0, keepdims=True))/(data.std(axis=0, keepdims=True)+1e-6)
    data = data.astype(float)
    total_size = data.shape[0]
    train_size = int(total_size * (1 - 0.1))
    keys = list(range(total_size))
    np.random.RandomState(seed).shuffle(keys)
    train_idx = keys[:train_size]
    test_idx = keys[train_size:]
    train_set = gen_index_train_tabular_dataset(data[train_idx], partial_targets[train_idx], ord_labels[train_idx])
    test_set = gen_index_test_tabular_dataset(data[test_idx], ord_labels[test_idx])
    return train_set, test_set