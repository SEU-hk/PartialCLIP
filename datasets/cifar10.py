from collections import defaultdict
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

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


class CIFAR10(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=None, train=train, transform=transform)


class CIFAR10_IR10(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform)

class CIFAR10_IR20(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.05, train=train, transform=transform)
        
class CIFAR10_IR50(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform)


class CIFAR10_IR100(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform)
        

class CIFAR10_IR150(IMBALANCECIFAR10):
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

