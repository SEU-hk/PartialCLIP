import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
import pickle
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models as vmodels
from datasets.inat2018 import iNaturalist2018_Augmentention

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224
from sklearn.preprocessing import OneHotEncoder
from scipy.special import comb

import datasets
from models import *
from datasets.imagenet_lt import ImageNet_Augmentention
from datasets.places_lt import Places_Augmentention
from datasets.cifar10 import *
from datasets.cifar100 import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator, compute_accuracy
from utils.templates import ZEROSHOT_TEMPLATES
from algorithms import *
# from LIFT2.utils import algorithms

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model




class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        # self.cfg['zsclip'] = self.cfg['zsclip'].to(self.device)
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None
        


    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.train_dataset = train_init_dataset
        
        self.num_classes = train_dataset.num_classes
        print("num_classes:", self.num_classes)
        self.num_instances = len(train_dataset.labels)
        self.cls_num_list = train_dataset.cls_num_list
        # print("cls_num_list:", self.cls_num_list)
        print(max(self.cls_num_list), min(self.cls_num_list))
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR20", "CIFAR100_IR50", "CIFAR100_IR100", "CIFAR100_IR150"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.ori_train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)
        
        ########################################################################################
        # Q1
        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            # cifar
            if cfg.dataset.startswith("CIFAR"):
                data, labels = train_dataset.data, torch.Tensor(train_dataset.labels).long()
            # imagenet, places, inaturalist2018
            else:
                data, labels = train_dataset.img_path, torch.Tensor(train_dataset.labels).long()
            # get original data and labels
            
            self.data = data
            self.targets = labels
            print(len(labels))
            
            if 0 < cfg.partial_rate < 1:
                partialY = self.fps(cfg.partial_rate)
            elif cfg.partial_rate == 0:
                partialY = self.uss()
            else:
                partialY = self.instance_dependent_generation()

            temp = torch.zeros(partialY.shape)
            temp[torch.arange(partialY.shape[0]), labels] = 1
            if torch.sum(partialY * temp) == partialY.shape[0]:
                print('partialY correctly loaded')
            else:
                print('inconsistent permutation')
            print('Average candidate num: ', partialY.sum(1).mean())
            
            self.test_loader = DataLoader(test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
            
            self.train_test_loader = DataLoader(train_test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
            
            partialY[torch.arange(partialY.shape[0]), labels] = 1
            
            if cfg.dataset == "ImageNet_LT":
                self.train_givenY = ImageNet_Augmentention(data, partialY.float(), labels.float(), transform_train, transform_plain)
            elif cfg.dataset == "Places_LT":
                self.train_givenY = Places_Augmentention(data, partialY.float(), labels.float(), transform_train, transform_plain)
            elif cfg.dataset.startswith("CIFAR"):
                self.train_givenY = CIFAR_Augmentation(data, partialY.float(), labels.float(), transform_train, transform_plain)
            else:
                self.train_givenY = iNaturalist2018_Augmentention(data, partialY.float(), labels.float(), transform_train, transform_plain)
            
            self.partialY = partialY
            
            self.train_loader = DataLoader(dataset=self.train_givenY, 
                batch_size=cfg.batch_size, shuffle=True,
                num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        ########################################################################################
        
        # Q2
        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=cfg.batch_size, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        # Q3
        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
            
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            
            if cfg['loss_type'] in ["DIRK", "ABLE", "PiCO"]:
                self.model = PeftModelFromCLIP_MLP(cfg, clip_model, num_classes)
                self.model.to(self.device)
                self.tuner = self.model.tuner
                self.neck = self.model.neck
                self.head = self.model.head
            
            else:
                self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
                self.model.to(self.device)
                self.tuner = self.model.tuner
                self.head = self.model.head


        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    
    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        print('Calculating uniform targets...')
        tempY = self.partialY.sum(dim=1).unsqueeze(1).repeat(1, self.partialY.shape[1])
        confidence = self.partialY.float()/tempY
        confidence = confidence.cuda()
        self.confidence = confidence 
        
        N = self.num_instances
        C = self.num_classes
        tensor = torch.randn(N, C)
        print(tensor.shape)
        algorithm_class = get_algorithm_class(cfg["loss_type"])
        # self.algorithm = algorithm_class(self.model, self.train_dataset.data.shape, self.partialY, cfg)
        self.algorithm = algorithm_class(self.model, tensor.shape, self.partialY, cfg)
        self.algorithm = self.algorithm.to(self.device)
            
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)


    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)
            
            
    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            self.algorithm.train()
            start = time.time()

            num_batches = len(self.train_loader)
            
            train_minibatches_iterator = iter(self.train_loader)
            
            with tqdm(total=num_batches, desc="Training", unit="batch") as pbar:
                for batch in train_minibatches_iterator:
                    batch_device = [item.to(self.device) for item in batch]
                    if cfg['loss_type'] in ['Solar', 'HTC']:
                        loss = self.algorithm.update(batch_device, epoch_idx)
                    else:
                        loss = self.algorithm.update(batch_device)
                        
                    # loss_meter.update(loss.item())
                    # batch_time.update(time.time() - start)
                    # print(f"time {batch_time.val:.3f} ({batch_time.avg:.3f})")
                    # print(f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})")
                    
                    pbar.update(1)
            
            if cfg['loss_type'] == 'Solar':
                self.algorithm.distribution_update(self.train_loader)
                
            end = time.time()
            print(f"Epoch: {epoch_idx}, Epoch time: {end - start}")  
              
            self.test()
                           
        torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)
        
        self.test()

        # Close writer
        self._writer.close()
        
    def fps(self, partial_rate=0.1):
        train_labels = self.targets
        if torch.min(train_labels) > 1:
            raise RuntimeError('testError')
        elif torch.min(train_labels) == 1:
            train_labels = train_labels - 1

        K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
        n = train_labels.shape[0]

        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0
        p_1 = partial_rate
        transition_matrix =  np.eye(K)
        transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
        print(transition_matrix)

        random_n = np.random.uniform(0, 1, size=(n, K))

        for j in range(n):  # for each instance
            for jj in range(K): # for each class 
                if jj == train_labels[j]: # except true class
                    continue
                if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                    partialY[j, jj] = 1.0

        print("Finish Generating Candidate Label Sets!\n")
        return partialY


    def instance_dependent_generation(self):
        def binarize_class(y):
            label = y.reshape(len(y), -1)
            enc = OneHotEncoder(categories='auto')
            enc.fit(label)
            label = enc.transform(label).toarray().astype(np.float32)
            label = torch.from_numpy(label)
            return label

        def create_model(ds, feature, c):
            from partial_models.resnet import resnet
            from partial_models.mlp import mlp_phi
            from partial_models.wide_resnet import WideResNet
            if ds in ['kmnist', 'fmnist']:
                model = mlp_phi(feature, c)
            elif ds in ['CIFAR10']:
                # model = resnet(depth=32, n_outputs=c)
                model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
            elif ds in ['CIFAR100']:
                # model = torchvision.models.wide_resnet50_2()
                # model.fc = nn.Linear(model.fc.in_features, c)
                model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
            else:
                pass
            return model

        with torch.no_grad():
            c = max(self.targets) + 1
            data = torch.from_numpy(self.data)
            y = binarize_class(self.targets.clone().detach().long())
            ds = self.cfg['dataset']

            f = np.prod(list(data.shape)[1:])
            batch_size = 2000
            rate = 0.4
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weight_path = f'./weights/{ds}.pt'
                
            model = create_model(ds, f, c).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            train_X, train_Y = data.to(device), y.to(device)

            train_X = train_X.permute(0, 3, 1, 2).to(torch.float32)
            train_p_Y_list = []
            step = train_X.size(0) // batch_size
            for i in range(0, step):
                outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
                train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
                partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
                partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
                partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
                partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
                partial_rate_array[partial_rate_array > 1.0] = 1.0
                m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
                z = m.sample()
                train_p_Y[torch.where(z == 1)] = 1.0
                train_p_Y_list.append(train_p_Y)
            train_p_Y = torch.cat(train_p_Y_list, dim=0)
            assert train_p_Y.shape[0] == train_X.shape[0]
        final_y = train_p_Y.cpu().clone()
        pn = final_y.sum() / torch.ones_like(final_y).sum()
        print("Partial type: instance dependent, Average Label: " + str(pn * 10))
        return train_p_Y.cpu()


    def uss(self): 
        train_labels = self.targets
        if torch.min(train_labels) > 1:
            raise RuntimeError('testError')
        elif torch.min(train_labels) == 1:
            train_labels = train_labels - 1
            
        K = torch.max(train_labels) - torch.min(train_labels) + 1
        n = train_labels.shape[0]
        cardinality = (2**K - 2).float()
        number = torch.tensor([comb(K, i+1) for i in range(K-1)]).float() # 1 to K-1 because cannot be empty or full label set, convert list to tensor
        frequency_dis = number / cardinality
        prob_dis = torch.zeros(K-1) # tensor of K-1
        for i in range(K-1):
            if i == 0:
                prob_dis[i] = frequency_dis[i]
            else:
                prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

        random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float() # tensor: n
        mask_n = torch.ones(n) # n is the number of train_data
        partialY = torch.zeros(n, K)
        partialY[torch.arange(n), train_labels] = 1.0
        
        temp_num_partial_train_labels = 0 # save temp number of partial train_labels
        
        for j in range(n): # for each instance
            for jj in range(K-1): # 0 to K-2
                if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                    temp_num_partial_train_labels = jj+1 # decide the number of partial train_labels
                    mask_n[j] = 0
                    
            temp_num_fp_train_labels = temp_num_partial_train_labels - 1
            candidates = torch.from_numpy(np.random.permutation(K.item())).long() # because K is tensor type
            candidates = candidates[candidates!=train_labels[j]]
            temp_fp_train_labels = candidates[:temp_num_fp_train_labels]
            
            partialY[j, temp_fp_train_labels] = 1.0 # fulfill the partial label matrix
        print("Finish Generating Candidate Label Sets!\n")
        return partialY

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
        
        # if self.feat_mean is not None:
        #     bias = self.model.head(self.feat_mean.unsqueeze(0)).detach()
        #     bias = F.softmax(bias, dim=1)
            
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                output, _ = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
                # if self.feat_mean is not None:
                #     output = output - torch.log(bias + 1e-9)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]


    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)

        if head_dict["weight"].shape == self.head.weight.shape:
            self.head.load_state_dict(head_dict, strict=False)
