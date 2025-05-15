import math
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.data = self.underlying_dataset.data[self.keys]
        self.partial_targets = self.underlying_dataset.partial_targets[self.keys]
    def __getitem__(self, key):
        weak_image, strong_image, partial_targets, ord_labels = self.underlying_dataset[self.keys[key]]
        return weak_image, strong_image, partial_targets, ord_labels, key
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def accuracy(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            batch_weights = torch.ones(len(x))
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def val_accuracy(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, x_strong, partial_y, y, _ in loader:
            x = x.to(device)
            partial_y = partial_y.to(device)
            y = y.to(device)
            p = network.predict(x)
            # print(f"Shape of x: {x.shape}")
            # print(f"Shape of p: {p.shape}")
            if p.size(1) == 1:
                correct += p.gt(0).eq(y).float().sum().item()
            else:
                correct += p.argmax(1).eq(y).float().sum().item()
            total += len(x)
    network.train()

    return correct / total


def val_covering_rate(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, x_strong, partial_y, _, _ in loader:
            x = x.to(device)
            partial_y = partial_y.to(device)
            p = network.predict(x)
            predicted_label = p.argmax(1)
            covering_per_example = partial_y[torch.arange(len(x)), predicted_label]
            correct += covering_per_example.sum().item()
            total += len(x)
    network.train()

    return correct / total

def val_approximated_accuracy(network, loader, device):
    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, x_strong, partial_y, _, _ in loader:
            x = x.to(device)
            partial_y = partial_y.to(device)
            batch_outputs = network.predict(x)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            label_confidence = temp_un_conf * partial_y
            base_value = label_confidence.sum(dim=1).unsqueeze(1).repeat(1, label_confidence.shape[1]) + 1e-12
            label_confidence = label_confidence / base_value
            predicted_label = batch_outputs.argmax(1)
            risk_mat = torch.ones_like(partial_y).float()
            risk_mat[torch.arange(len(x)), predicted_label] = 0
            correct += len(x) - (risk_mat * label_confidence).sum().item()
            total += len(x)
    network.train()
    return correct / total

def topk_accuracy(network, loader, device, topk=5):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = network.predict(x)
            batch_weights = torch.ones(len(x))
            batch_weights = batch_weights.to(device)

            _, pred = output.topk(topk, 1, True, True)
            pred = pred.t()
            correct_mat = pred.eq(y.view(1, -1).expand_as(pred))
            correct_k_curr = correct_mat[:topk].reshape((-1, )).float().sum(0, keepdim=True)
            correct += correct_k_curr.item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total