import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
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

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224
from sklearn.preprocessing import OneHotEncoder
from scipy.special import comb

import datasets
from models import *
from datasets.cifar10 import *
from datasets.cifar100 import *
from datasets.places_lt import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.pll_loss import *
from utils.evaluator import Evaluator, compute_accuracy
from utils.templates import ZEROSHOT_TEMPLATES
from utils.visual import *
from algorithms import *
from utils.util import *
from utils.candidate_set_generation import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None
        


    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution

        if cfg.backbone.startswith("IN21K"):
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
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

        if cfg.dataset.startswith("CIFAR"):
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        self.ori_train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)
        
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
            
            self.test_loader = DataLoader(test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
            
            self.train_test_loader = DataLoader(train_test_dataset,
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)
            
            partialY[torch.arange(partialY.shape[0]), labels] = 1

            if cfg.pre_filter == True:
                partialY, data, labels = pre_filter(cfg, partialY, labels)
            
            if cfg.dataset.startswith("CIFAR"):
                self.train_givenY = CIFAR_Augmentation(data, partialY.float(), labels.float(), transform_train, transform_plain)
            else:
                self.train_givenY = Places_Augmentention(data, partialY.float(), labels.float(), transform_train, transform_plain)
            
            self.partialY = partialY
            
            self.train_loader = DataLoader(dataset=self.train_givenY, 
                batch_size=cfg.batch_size, shuffle=True,
                num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
        
        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=cfg.batch_size, sampler=None, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

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
            self.model = PeftModelFromCLIP_HTC(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head1 = self.model.head1
            self.head = self.model.head


        else:
            print(f"Loading backbone: {cfg.backbone}")
            vit_model = load_model_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT_HTC(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
            self.head1 = self.model.head1

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
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
        print("Turning on gradients in the head1")
        for name, param in self.head1.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        head1_params = sum(p.numel() for p in self.head1.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        print(f"Head1 params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()},
                                      {"params": self.head1.parameters()}],
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

        self.head1.apply_weight(text_features)
        self.head.apply_weight(text_features)
 
            
    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()
        best_acc = 0
        
        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            self.algorithm.train()
            start = time.time()

            num_batches = len(self.train_loader)
            
            train_minibatches_iterator = iter(self.train_loader)
            
            self.algorithm.emp_dist_head = self.algorithm.confidence.sum(0) / self.algorithm.confidence.sum()
            self.algorithm.emp_dist_head = self.algorithm.emp_dist_head.to(self.device)
            
            with tqdm(total=num_batches, desc="Training", unit="batch") as pbar:
                for batch in train_minibatches_iterator:
                    batch_device = [item.to(self.device) for item in batch]
                    loss = self.algorithm.update(batch_device, epoch_idx)
                    
                    pbar.update(1)
                    
            
            if cfg['loss_type'] == 'HTC':
                confidence = self.algorithm.confidence
                print(confidence.shape)
                column_sums = confidence.sum(dim=0, keepdim=True)
                normalized_sums = column_sums / column_sums.sum()
                print(normalized_sums.tolist()) 
                
            end = time.time()
            print(f"Epoch: {epoch_idx}, Epoch time: {end - start}")  
              
            acc = self.test_HTC()
            
            best_acc = max(best_acc, acc)
            
            print("Best_accuracy:", best_acc)
 
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
        
        self.test_HTC()

        # Close writer
        self._writer.close()
    
    
    @torch.no_grad()
    def test_HTC(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        if self.head1 is not None:
            self.head1.eval()
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

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                logit_head, logit_tail, _ = self.model(image)
                output = self.model.ensemble(logit_head, logit_tail, self.algorithm.loss_fn.get_distribution())
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        acc = results["accuracy"]
         
        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return acc


    def save_model(self, directory):
        # Initialize checkpoint dictionary
        checkpoint = {}
        
        # Save head parameters (always saved)
        head_dict = self.head.state_dict()
        
        # Save tuner parameters if it exists
        if self.tuner is not None:
            tuner_dict = self.tuner.state_dict()
            checkpoint["tuner"] = tuner_dict
        else:
            print("Warning: tuner is None. Skipping tuner state_dict save.")
        
        checkpoint["head"] = head_dict
        
        # Remove 'module.' prefix from state_dict keys (for DataParallel models)
        for key in checkpoint.keys():
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]  # Remove 'module.' prefix
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict
        
        # Save model checkpoint
        os.makedirs(directory, exist_ok=True)  # Ensure directory exists
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'Checkpoint not found at "{load_path}"')
        
        checkpoint = torch.load(load_path, map_location=self.device)
        head_dict = checkpoint["head"]
        
        # Load tuner weights if tuner exists and checkpoint contains tuner
        if self.tuner is not None and "tuner" in checkpoint:
            tuner_dict = checkpoint["tuner"]
            print(f"Loading tuner weights from {load_path}")
            self.tuner.load_state_dict(tuner_dict, strict=False)
        else:
            print("Warning: tuner is None or checkpoint does not contain tuner. Skipping tuner load.")
        
        # Load head weights with shape validation
        if head_dict["weight"].shape == self.head.weight.shape:
            print(f"Loading head weights from {load_path}")
            self.head.load_state_dict(head_dict, strict=False)
        else:
            print(f"Head weight shape mismatch. Expected {self.head.weight.shape}, got {head_dict['weight'].shape}")
