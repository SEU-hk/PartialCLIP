import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

from .clip_text import CLIP_Text
from finetune.peft_vit import *
from finetune.peft_rn import Peft_RN, RN_Tuner
from head.classifiers import *

class ViT(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit, image_features
    
    
class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        # self.dtype = clip_model.dtype
        self.dtype = torch.float32

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit, image_features


class LinearProbingCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype
        feat_dim = 512
        # self.linear_probe = LinearProbe(feat_dim, num_classes)
        self.head = eval(cfg.classifier)(feat_dim, num_classes, self.dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.head(image_features)
        return logit, image_features
    

class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()

        if "ViT" in cfg.backbone:
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_ViT(clip_model.visual, cfg)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
        elif "RN" in cfg.backbone:
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        print(f"Feat dim: {feat_dim}")
        dtype = self.image_encoder.dtype

        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)

    

class PeftModelFromCLIP_MLP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()

        self.cfg = cfg
        self.clip_model = clip_model
        self.num_classes = num_classes
        
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = Peft_ViT_MLP(clip_model.visual)
        self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)

        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype

        self.neck = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim))
        
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def forward(self, image, use_tuner=True, return_feature=False, return_logits=True):
        tuner = self.tuner if use_tuner else None
        neck = self.neck if not return_feature else None
        head = self.head if return_logits else None
        return self.image_encoder(image, tuner, neck, head)
    
    
    
class PeftModelFromCLIP_HTC(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()

        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = Peft_ViT_HTC(clip_model.visual)
        self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)

        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype

        self.head1 = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.attention = nn.Linear(num_classes * 2, 2)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def ensemble(self, logit_head, logit_tail, distribution):
        p1, p2 = F.softmax(logit_head - torch.log(distribution+1e-5), dim=1), F.softmax(logit_tail, dim=1)
        weights = F.softmax(F.leaky_relu(self.attention(torch.cat([p1, p2], dim=1))), dim=1)
        w1, w2 = torch.split(F.normalize(weights, p=2), 1, dim=1)
        pred = F.softmax(w1 * p1 + w2 * p2, dim=1)
        return pred
    
    def forward(self, image, use_tuner=True, head=True, tail=True):
        tuner = self.tuner if use_tuner else None
        head1 = self.head1 if head else None
        head2 = self.head if tail else None
        return self.image_encoder(image, tuner, head1, head2)
    
    
class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        self.image_encoder = Peft_ViT(vit_model, cfg)
        self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)
    

class PeftModelFromViT_MLP(nn.Module):
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        self.image_encoder = Peft_ViT_MLP(vit_model)
        self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim))

    def forward(self, image, use_tuner=True, return_feature=True, return_logits=True):
        tuner = self.tuner if use_tuner else None
        neck = self.neck if return_feature else None
        head = self.head if return_logits else None
        return self.image_encoder(image, tuner, neck, head)
    
     
class PeftModelFromViT_HTC(nn.Module):
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        self.image_encoder = Peft_ViT_HTC(vit_model)
        self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.head1 = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)
        self.attention = nn.Linear(num_classes * 2, 2)

    def ensemble(self, logit_head, logit_tail, distribution):
        p1, p2 = F.softmax(logit_head - torch.log(distribution+1e-5), dim=1), F.softmax(logit_tail, dim=1)
        weights = F.softmax(F.leaky_relu(self.attention(torch.cat([p1, p2], dim=1))), dim=1)
        w1, w2 = torch.split(F.normalize(weights, p=2), 1, dim=1)
        pred = F.softmax(w1 * p1 + w2 * p2, dim=1)
        return pred
    
    def forward(self, image, use_tuner=True, head=True, tail=True):
        tuner = self.tuner if use_tuner else None
        head1 = self.head1 if head else None
        head2 = self.head if tail else None
        return self.image_encoder(image, tuner, head1, head2)
    
    
    
    
