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
from datasets.cifar10 import *
from datasets.plcifar10 import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.pll_loss import *
from utils.evaluator import Evaluator, compute_accuracy
from utils.templates import ZEROSHOT_TEMPLATES
from utils.visual import *
from utils.algorithms import *
from utils.misc import *

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
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg)
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

        if cfg.dataset == "PLCIFAR10_Aggregate":
            train_dataset = PLCIFAR10_Aggregate(root)
            train_init_dataset = PLCIFAR10_Aggregate(root)
            train_test_dataset = PLCIFAR10_Aggregate(root)
            
        elif cfg.dataset == "PLCIFAR10_Vaguest":
            train_dataset = PLCIFAR10_Vaguest(root)
            train_init_dataset = PLCIFAR10_Vaguest(root)
            train_test_dataset = PLCIFAR10_Vaguest(root)
        
        test_dataset = CIFAR10(root, train=False, transform=transform_test)

        self.train_dataset = train_init_dataset

        self.partialY = train_dataset.partial_targets
        self.partialY = torch.from_numpy(self.partialY)
        self.num_classes = train_dataset.num_classes
        self.num_instances = len(train_dataset.data)
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                              
        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            val_dataset, train_dataset = split_dataset(train_dataset, int(self.num_instances*0.1), seed_hash(0))
            
            self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
            
            self.test_loader = DataLoader(test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=False, drop_last=False)
            
            self.train_test_loader = DataLoader(train_test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=False)
            
            self.train_loader = DataLoader(dataset=train_dataset, 
                batch_size=cfg.batch_size, shuffle=True,
                num_workers=cfg.num_workers, pin_memory=False, drop_last=True)
        ########################################################################################
        
        # Q3
        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=False)

        self.test_loader = DataLoader(test_dataset,
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=False, drop_last=False)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size


    def build_model(self):
        cfg = self.cfg
        classnames = ['airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = 10

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
            
            if cfg['loss_type'] in ["ABLE"]:
                self.model = PeftModelFromCLIP_MLP(cfg, clip_model, num_classes)
                self.model.to(self.device)
                self.tuner = self.model.tuner
                self.neck = self.model.neck
                self.head = self.model.head
            
            elif cfg['loss_type'] in ["PiCO"]:
                print(f"Loading CLIP Again (backbone: {cfg.backbone})")
                clip_model1 = load_clip_to_cpu(cfg.backbone, cfg.prec)
                self.model = PeftModelFromCLIP_MLP(cfg, clip_model, num_classes)
                self.model.to(self.device)
                self.tuner = self.model.tuner
                self.neck = self.model.neck
                self.head = self.model.head
                
                self.model1 = PeftModelFromCLIP_MLP(cfg, clip_model1, num_classes)
                self.model1.to(self.device)
                self.tuner1 = self.model1.tuner
                self.neck1 = self.model1.neck
                self.head1 = self.model1.head
                
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
            elif cfg.init_head == "vit":
                self.init_head_vit()
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
            
        if cfg['loss_type'] in ["DIRK", "ABLE",]:
            print("Turning on gradients in the neck")
            for name, param in self.neck.named_parameters():
                param.requires_grad_(True)
                
        elif cfg['loss_type'] in ["PiCO"]:
            print("Turning off gradients in the model1")
            for name, param in self.model1.named_parameters():
                param.requires_grad_(False)     

            print("Turning on gradients in the head1")
            for name, param in self.head1.named_parameters():
                param.requires_grad_(True)
            print("Turning on gradients in the tuner1")
            for name, param in self.tuner1.named_parameters():
                param.requires_grad_(True)
            print("Turning on gradients in the neck")
            for name, param in self.neck.named_parameters():
                param.requires_grad_(True)
            print("Turning on gradients in the neck1")
            for name, param in self.neck1.named_parameters():
                param.requires_grad_(True)            

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    
    def build_criterion(self):
        cfg = self.cfg

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
        if cfg['loss_type'] in ["PiCO"]:
            self.algorithm = algorithm_class([self.model, self.model1], tensor.shape, self.partialY, cfg)
        else:
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

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        running_times = []
        
        
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
                    
                    pbar.update(1)
            
            val_acc = val_accuracy(self.algorithm, self.val_loader, self.device) 
            print(f'val_accuracy: {val_acc}')
            val_cr = val_covering_rate(self.algorithm, self.val_loader, self.device)
            print(f'val_covering_rate: {val_cr}')
            val_approximated_acc = val_approximated_accuracy(self.algorithm, self.val_loader, self.device)
            print(f'val_approximated_acc: {val_approximated_acc}')
            
            if cfg['loss_type'] == 'Solar':
                self.algorithm.distribution_update(self.train_loader)
                
            end = time.time()
            print(f"Epoch: {epoch_idx}, Epoch time: {end - start}") 
             
            running_times.append(end - start)
            
            self.test()
                           
        torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        mean_running_time = sum(running_times) / (num_epochs * 781.25)
        print(f"Running Time: {mean_running_time:.2f}")
        
        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)
        
        self.test()

        # Close writer
        self._writer.close()
        

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "val":
            print(f"Evaluate on the val set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
        
        if self.cfg['loss_type'] == 'RECORDS':
            bias = self.model.head(self.algorithm.criterion.feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
            
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
                if self.cfg['loss_type'] == 'RECORDS':
                    output = output - torch.log(bias + 1e-9)
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


    @torch.no_grad()
    def test1(self, mode="test"):
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
        
            
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            # print(image.size())
            # _bsz, _c, _h, _w = image.size()
            # _ncrops = 1
            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            outputs = []
            labels = []
        
            if _ncrops <= 8:
                output, _ = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            outputs.append(output)
            labels.append(label)
            
            self.evaluator.process(output, label)

        # 合并outputs列表中的所有输出
        outputs = torch.cat(outputs, dim=0)

        mask_dir = '/home/hekuang/LIFT-main/LIFT2/confidence'
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        # 保存文件到mask_dir目录下，文件名为self.cfg.dataset.pth
        torch.save(outputs, os.path.join(mask_dir, self.cfg.dataset + '.pth'))
        
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
