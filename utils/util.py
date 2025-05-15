import timm
import torch
import numpy as np
from torch import nn

import math
import numpy as np
from scipy.special import comb
import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from clip import clip
from timm.models.vision_transformer import vit_base_patch16_384, vit_base_patch16_224, vit_large_patch16_224


def generate_instancedependent_candidate_labels(model, train_X, train_Y,RATE=0.4):
    with torch.no_grad():
        k = int(torch.max(train_Y) - torch.min(train_Y) + 1)
        n = train_Y.shape[0]
        model = model.cuda()
        train_Y = torch.nn.functional.one_hot(train_Y, num_classes=k)
        avg_C = 0
        partialY_list = []
        rate, batch_size = RATE, 2000
        step = math.ceil(n / batch_size)

        for i in range(0, step):
            b_end = min((i + 1) * batch_size, n)

            train_X_part = train_X[i * batch_size: b_end].cuda()

            outputs = model(train_X_part)

            train_p_Y = train_Y[i * batch_size: b_end].clone().detach()

            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_p_Y == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0

            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()

            train_p_Y[torch.where(z == 1)] = 1.0
            partialY_list.append(train_p_Y)

        partialY = torch.cat(partialY_list, dim=0).float()

        assert partialY.shape[0] == train_X.shape[0]

    avg_C = torch.sum(partialY) / partialY.size(0)

    return partialY



def load_model_to_cpu(backbone_name, prec):
    """
    Loads a vision model to CPU with specified precision.
    
    Args:
        backbone_name (str): Name of the backbone model.
        prec (str): Precision mode, supports "fp16", "fp32", "amp".
        
    Returns:
        torch.nn.Module: Loaded model with specified precision.
    """
    # Model configuration mapping
    model_configs = {
        # META-ViT series configurations
        'META-ViT-B/16': {
            'creator': lambda pretrained: timm.create_model('vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b.bin',
            'loader': 'timm'
        },
        'META-ViT-B/32': {
            'creator': lambda pretrained: timm.create_model('vit_base_patch32_clip_quickgelu_224.metaclip_2pt5b', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_base_patch32_clip_quickgelu_224.metaclip_2pt5b.bin',
            'loader': 'timm'
        },
        'META-ViT-L/14': {
            'creator': lambda pretrained: timm.create_model('vit_large_patch14_clip_quickgelu_224.metaclip_2pt5b', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_large_patch14_clip_quickgelu_224.metaclip_2pt5b.bin',
            'loader': 'timm'
        },
        # SIGLIP series configurations
        'SIGLIP-ViT-B/16': {
            'creator': lambda pretrained: timm.create_model('vit_base_patch16_siglip_224.webli', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_base_patch16_siglip_224.webli.bin',
            'loader': 'timm'
        },
        'SIGLIP2-ViT-B/16': {
            'creator': lambda pretrained: timm.create_model('vit_base_patch16_siglip_224.webli', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_base_patch16_siglip_224.v2_webli.bin',
            'loader': 'timm'
        },
        'SIGLIP-ViT-S/14': {
            'creator': lambda pretrained: timm.create_model('vit_so400m_patch14_siglip_224.webli', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_so400m_patch14_siglip_224.webli.bin',
            'loader': 'timm'
        },
        'SIGLIP2-ViT-S/14': {
            'creator': lambda pretrained: timm.create_model('vit_so400m_patch14_siglip_224.webli', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_so400m_patch14_siglip_224.v2_webli.bin',
            'loader': 'timm'
        },
        # OpenCLIP series configurations
        'ViT-B-16': {
            'creator': lambda: open_clip.create_model_and_transforms('ViT-B-16', pretrained='ckpt/vit_base_patch16_clip_224.laion400m_e31.bin')[0],
            'loader': 'open_clip'
        },
        'ViT-B-32': {
            'creator': lambda: open_clip.create_model_and_transforms('ViT-B-32', pretrained='ckpt/vit_base_patch32_clip_224.laion400m_e31.bin')[0],
            'loader': 'open_clip'
        },
        'ViT-L-14': {
            'creator': lambda: open_clip.create_model_and_transforms('ViT-L-14', pretrained='ckpt/vit_large_patch14_clip_224.laion400m_e31.bin')[0],
            'loader': 'open_clip'
        },
        # IN21K-ViT series configurations
        'IN21K-ViT-B/16': {
            'creator': lambda pretrained: timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_base_patch16_224.augreg_in1k.bin',
            'loader': 'timm'
        },
        'IN21K-ViT-B/16@384px': {
            'creator': lambda: vit_base_patch16_384(pretrained=True),
            'loader': 'torchvision'
        },
        'IN21K-ViT-L/16': {
            'creator': lambda pretrained: vit_large_patch16_224(pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_large_patch16_224.bin',
            'loader': 'torchvision'
        },
        'IN21K-ViT-S/16': {
            'creator': lambda pretrained: timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_small_patch16_224.augreg_in1k.bin',
            'loader': 'timm'
        },
        'IN21K-ViT-T/16': {
            'creator': lambda pretrained: timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=pretrained),
            'ckpt_path': 'ckpt/vit_tiny_patch16_224.augreg_in21k.bin',
            'loader': 'timm'
        }
    }
    
    # Check if the model name is supported
    if backbone_name not in model_configs:
        raise ValueError(f"Unsupported model name: {backbone_name}")
    
    # Get model configuration
    config = model_configs[backbone_name]
    
    try:
        # Create model with specified precision
        if config['loader'] == 'timm':
            # Try to load pretrained weights from internet first
            try:
                model = config['creator'](pretrained=True)
                print(f"Successfully loaded pretrained weights for {backbone_name} from internet.")
            except Exception as e:
                print(f"Failed to load pretrained weights for {backbone_name} from internet: {str(e)}")
                print(f"Loading weights from local path: {config['ckpt_path']}")
                model = config['creator'](pretrained=False)
                state_dict = torch.load(config['ckpt_path'], map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
        elif config['loader'] == 'torchvision' and 'ckpt_path' in config:
            # For torchvision models with local weights
            model = config['creator'](pretrained=False)
            state_dict = torch.load(config['ckpt_path'], map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        else:
            # For models that don't require manual weight loading
            model = config['creator']()
        
        # Set model to evaluation mode
        model = model.eval()
        
        # Set precision
        assert prec in ["fp16", "fp32", "amp"]
        if prec == "fp16":
            # Convert model to half precision
            model.half()
        elif prec == "fp32" or prec == "amp":
            # Convert model to full precision (default for CLIP is fp16)
            model.float()
            
        return model
        
    except Exception as e:
        print(f"Error loading model {backbone_name}: {str(e)}")
        raise
    
    
def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    if backbone_name[-3] == '-':
        backbone_name = backbone_name[:-3] + '/' + backbone_name[-2:]
        
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