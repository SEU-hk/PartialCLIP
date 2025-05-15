import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

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

import datasets
from models import *
from datasets.imagenet_lt import *
from datasets.places_lt import *
from datasets.cifar10 import *
from datasets.cifar100 import *

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

long_tailed = ["CIFAR10_IR100", "CIFAR10_IR150", "CIFAR10_IR200", "CIFAR10_IR250", "CIFAR100_IR10", "CIFAR100_IR20", "CIFAR100_IR50", "CIFAR100_IR100", "CIFAR100_IR150", "Places_LT", "ImageNet_LT"]

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

        
        if not (cfg.test_train or cfg.test_only):
            if cfg.dataset == 'fgvc100':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames = load_fgvc100(cfg, transform_train, transform_test)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'cub200':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames = load_cub200(cfg, transform_train, transform_test)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'cars196':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames = load_cars196(cfg, transform_train, transform_test)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'dogs120':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames = load_dogs120(cfg, transform_train, transform_test)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'CIFAR10_ID':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames = load_cifar10_id(cfg, transform_train, transform_test)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'CIFAR100_ID':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames = load_cifar100_id(cfg, transform_train, transform_test)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'Places_LT':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames, self.cls_num_list = load_places_lt(cfg, transform_train, transform_test, transform_plain)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset == 'ImageNet_LT':
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames, self.cls_num_list = load_imagenet_lt(cfg, transform_train, transform_test, transform_plain)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset.startswith('CIFAR100'):
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames, self.cls_num_list = load_cifar100_lt(cfg, transform_train, transform_test, transform_plain)
                print("num_classes:", self.num_classes)
                
            elif cfg.dataset.startswith('CIFAR10'):
                self.train_loader, self.train_test_loader, self.partialY, self.test_loader, self.num_instances, self.num_classes, self.classnames, self.cls_num_list = load_cifar10_lt(cfg, transform_train, transform_test, transform_plain)
                print("num_classes:", self.num_classes)
        
        if cfg.dataset in long_tailed:
            if cfg.dataset.startswith("CIFAR"):
                split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
            else:
                split_cls_num_list = self.cls_num_list
            self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
            self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
            self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]
        
        else:
            self.many_idxs = None
            self.med_idxs = None
            self.few_idxs = None
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

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


        elif cfg.linear_probing:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading LinearProbingCLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = LinearProbingCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = None
            self.head = self.model.head
            
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)
        
        
        elif cfg.backbone.startswith("CLIP") or cfg.backbone.startswith("ViT"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            
            if cfg['loss_type'] in ["ABLE"]:
                self.model = PeftModelFromCLIP_MLP(cfg, clip_model, num_classes)
                self.neck = self.model.neck
            else:
                self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
                
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        else:
            print(f"Loading backbone: {cfg.backbone}")
            vit_model = load_model_to_cpu(cfg.backbone, cfg.prec)
            
            if cfg['loss_type'] in ["ABLE"]:
                self.model = PeftModelFromViT_MLP(cfg, vit_model, num_classes)
                self.neck = self.model.neck
            else:
                self.model = PeftModelFromViT(cfg, vit_model, num_classes)
                
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

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
        
        if self.tuner is not None:
            print("Turning on gradients in the tuner")
            for name, param in self.tuner.named_parameters():
                param.requires_grad_(True)
            
        print("Turning on gradients in the head") 
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)
            
        if cfg['loss_type'] in ["ABLE"]:
            print("Turning on gradients in the neck")
            for name, param in self.neck.named_parameters():
                param.requires_grad_(True)
               

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total params: {total_params}")
        
        if cfg.linear_probing == False:
            tuned_params = sum(p.numel() for p in self.tuner.parameters())
            print(f"Tuned params: {tuned_params}")
            
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Head params: {head_params}")
        
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        if cfg.linear_probing == False:
            params = [{"params": self.tuner.parameters()}, {"params": self.head.parameters()}]
        else:
            params = [{"params": self.head.parameters()}]

        if cfg.loss_type == "ABLE":
            params.append({"params": self.neck.parameters()})

        self.optim = torch.optim.SGD(params,
                                    lr=cfg.lr, 
                                    weight_decay=cfg.weight_decay, 
                                    momentum=cfg.momentum)
            
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

        if cfg.linear_probing == False:
            if cfg.backbone.startswith("CLIP-ViT") or cfg.backbone.startswith("ViT"):
                text_features = text_features @ self.model.image_encoder.proj.t()
                text_features = F.normalize(text_features, dim=-1)

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

        num_epochs = cfg.num_epochs
        running_times = []
        best_acc = 0
        
        for epoch_idx in range(num_epochs):
            if self.tuner is not None:
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
            
            if cfg['loss_type'] == 'Solar':
                self.algorithm.distribution_update(self.train_loader)
                print(self.algorithm.emp_dist.tolist())
                
            end = time.time()
            print(f"Epoch: {epoch_idx}, Epoch time: {end - start}") 
             
            running_times.append(end - start)
            
            acc = self.test()
            best_acc = max(best_acc, acc)
            
            print("Best_accuracy:", best_acc)
                           
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
        
        acc = self.test()
        best_acc = max(best_acc, acc)
        
        print("Best_accuracy:", best_acc)

        # Close writer
        self._writer.close()

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

        return results["accuracy"]

    @torch.no_grad()
    def test_confidence(self, mode="train"):
        """
        Generate and save zero-shot confidence scores on training data
        
        Args:
            mode: Dataset mode, default "train" uses training dataset
        """
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
                    
        # Create save directory and path
        save_dir = "confidence"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.cfg.dataset}.pth")
        
        # Select dataset loader
        if mode == "train":
            print(f"Generating zero-shot confidence on train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Generating zero-shot confidence on test set")
            data_loader = self.test_loader
        
        # Initialize storage
        all_outputs = []
        all_labels = []
                
        # Process batches
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz,  _c, _h, _w = image.size()

            # Forward pass with multi-crop processing
            output, _ = self.model(image)
            # print(output.size())
            # output = output.view(_bsz, _ncrops, -1).mean(dim=1)

            # Collect batch results
            all_outputs.append(output.cpu())
            all_labels.append(label.cpu())
        
        # Concatenate results
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Save results to file
        torch.save(all_outputs, save_path)
        print(f"Zero-shot raw outputs saved to {save_path}")
        
        return

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
