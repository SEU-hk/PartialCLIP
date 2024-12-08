import os
from .lt_data import LT_Dataset
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .randaugment import RandomAugment

class iNaturalist2018(LT_Dataset):
    category_method = "name"
    categories_json = "./datasets/iNaturalist18/categories.json"
    train_txt = "./datasets/iNaturalist18/iNaturalist18_train.txt"
    test_txt = "./datasets/iNaturalist18/iNaturalist18_val.txt"

    def __init__(self, root, train=True, transform=None, partial_rate=0.1):
        super().__init__(root, train, transform)

        id2cname, cname2lab = self.read_category_info()

        self.names = []
        self.labels = []
        with open(self.txt) as f:
            for line in f:
                path, id = line.split()
                _name = id2cname[int(id)]
                self.names.append(_name)
                self.labels.append(cname2lab[_name])

        self.classnames = self.get_classnames()
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

        if train:  # 只有在训练模式下生成偏标记数据
            self.partialY = self.generate_partial_labels(
                torch.tensor(self.labels), partial_rate
            )

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        name = self.names[index]
        if hasattr(self, 'partialY'):
            partial_label = self.partialY[index]
            return image, label, name, partial_label
        else:
            return image, label, name

    @classmethod
    def read_category_info(self):
        with open(self.categories_json, "rb") as file:
            category_info = json.load(file)
        
        id2cname = {}
        for id, info in enumerate(category_info):
            cname = info[self.category_method]
            id2cname[id] = cname

        cnames_unique = sorted(set(id2cname.values()))
        cname2lab = {c: i for i, c in enumerate(cnames_unique)}
        return id2cname, cname2lab

    def get_classnames(self):
        container = set()
        for label, name in zip(self.labels, self.names):
            container.add((label, name))
        mapping = {label: classname for label, classname in container}
        classnames = [mapping[label] for label in sorted(mapping.keys())]
        return classnames

    def generate_partial_labels(self, train_labels, partial_rate=0.1):
        # 检查标签是否从0开始，如果不是，则减去1
        if torch.min(train_labels) > 0:
            train_labels = train_labels - 1

        # 计算类别数量K和样本数量n
        K = int(torch.max(train_labels) + 1)  # 由于标签从0开始，所以直接加1
        n = train_labels.shape[0]

        # 初始化偏标记矩阵
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0  # 设置真实标签位置为1

        # 创建转移矩阵，对角线元素为1，其余元素为partial_rate
        transition_matrix = np.eye(K)
        transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate

        print('==> Transition Matrix:')
        print(transition_matrix)

        # 生成随机数矩阵，用于确定是否设置偏标记
        random_n = np.random.uniform(0, 1, size=(n, K))

        # 应用转移矩阵生成偏标记
        for j in range(n):  # 对每个实例
            partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

        print("Finish Generating Candidate Label Sets!\n")
        return partialY

class ImageNet_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels, con=True):
        """
        Args:
            images: images
            given_label_matrix: PLL candidate labels
            true_labels: GT labels
            con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        """
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.con = con

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        if self.con:
            each_image_w = self.weak_transform(self.images[index])
            each_image_s = self.strong_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            
            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            each_image_w = self.weak_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_label, each_true_label, index

class iNaturalist2018_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels, con=True):
        """
        Args:
            images: images
            given_label_matrix: PLL candidate labels
            true_labels: GT labels
            con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        """
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.con = con

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        if self.con:
            each_image_w = self.weak_transform(self.images[index])
            each_image_s = self.strong_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            
            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            each_image_w = self.weak_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_label, each_true_label, index
        
        
class iNaturalist2018_Kingdom(iNaturalist2018):
    category_method = "kingdom"

class iNaturalist2018_Phylum(iNaturalist2018):
    category_method = "phylum"

class iNaturalist2018_Class(iNaturalist2018):
    category_method = "class"

class iNaturalist2018_Order(iNaturalist2018):
    category_method = "order"

class iNaturalist2018_Family(iNaturalist2018):
    category_method = "family"

class iNaturalist2018_Genus(iNaturalist2018):
    category_method = "genus"

class iNaturalist2018_Species(iNaturalist2018):
    category_method = "name"
