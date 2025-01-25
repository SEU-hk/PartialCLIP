# UPB: A-Unified-Partial-label-learning-Benchmark

This is the source code for the paper.

<p align="center">
  <a href="#-introduction">🎉Introduction</a> •
  <a href="#-methods-reproduced">🌟Methods Reproduced</a> •
  <a href="#-reproduced-results">📝Reproduced Results</a> <br />
  <a href="#%EF%B8%8F-how-to-use">☄️How to Use</a> •
  <a href="#-acknowledgments">👨‍🏫Acknowledgments</a> •
  <a href="#-contact">🤗Contact</a>
</p>

---

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/PILOT-v1.0-darkcyan"></a>
  <a href='https://arxiv.org/abs/2309.07117'><img src='https://img.shields.io/badge/Arxiv-2309.07117-b31b1b.svg?logo=arXiv'></a>
  <a href=""><img src="https://img.shields.io/github/stars/sun-hailong/LAMDA-PILOT?color=4fb5ee"></a>
  <a href=""><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsun-hailong%2FLAMDA-PILOT&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false"></a>
  <a href=""><img src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
  <a href=""><img src="https://img.shields.io/github/last-commit/sun-hailong/LAMDA-PILOT?color=blue"></a>
</p>

## 🎉 Introduction

Welcome to (**UPB**): a unified partial label learning benchmark for classification. The UPB framework proposed by us has a clear structure. It integrates the state-of-the-art (SOTA) algorithms of (**PLL**) (Partial Label Learning), (**LT-PLL**) (Long-tailed Partial Label Learning), and (**IDPLL**)IDPLL (Instance-dependent Partial Label Learning), and provides a unified interface. Moreover, its code has been open-sourced on GitHub, allowing new methods and datasets to be added easily. 


## 📰 What's New
- [2024-12]🌟 Add [Stanford Dogs120](http://vision.stanford.edu/aditya86/ImageNetDogs/), [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Stanford Cars196](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) and [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) IDPLL datasets!
- [2024-12]🌟 Add SoTa *IDPLL* baselines, including [ABLE](https://arxiv.org/abs/2209.10365)(**IJCAI 2022**), [IDGP](https://arxiv.org/abs/2204.03845)(**ICLR 2023**). [POP](https://arxiv.org/abs/2206.00830)(**ICLR 2023**)!!
- [2024-11]🌟 Add [Places](http://places2.csail.mit.edu/download.html), [ImageNet](http://image-net.org/index) LT-PLL datasets.!
- [2024-11]🌟 Add SoTa *LT-PLL* baselines, including [Solar](https://arxiv.org/abs/2209.10365)(**ICLR 2022**), [HTC](https://arxiv.org/pdf/2007.08929)(**AAAI 2024**)!
- [2024-10]🌟 Add SoTa *PLL* baselines, including [CRDPLL](https://palm.seu.edu.cn/zhangml/files/ICML'22a.pdf)(**ICML 2022**), [PiCO](https://arxiv.org/pdf/2007.08929)(**ICLR 2022**), [ABS-MAE ABS-GCE](https://openreview.net/pdf?id=qqdXHUGec9h)(**TPAMI 2023**)!
- [2024-10]🌟 Add [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and their long-tailed versions.
- [2024-10]🌟 Add SoTa *PLL* baselines, including [PRODEN](https://arxiv.org/abs/2002.08053)(**ICML 2020**), [CC](https://arxiv.org/abs/2007.08929)(**NeurIPS 2020**), [LWS](https://arxiv.org/abs/2106.05731)(**ICML 2021**), [CAVL](https://openreview.net/pdf?id=qqdXHUGec9h)(**ICLR 2022**)!
- [2024-09]🌟 Initial version of UPB is released.
- [2024-09]🌟 Fistly, conduct experiments on [RECORDS](https://arxiv.org/abs/2302.05080)(*LT-PLL*)(**ICLR 2023**) with UPB!

## 🌟 Methods Reproduced

**PLL**

- `PRODEN`: Progressive Identification of True Labels for Partial-Label Learning. ICML 2020 [[paper](https://arxiv.org/abs/2002.08053)]
- `CC`: Provably Consistent Partial-Label Learning. NeurIPS 2020 [[paper](https://arxiv.org/abs/2007.08929)]
- `LWS`: Leveraged Weighted Loss for Partial Label Learning. ICML 2021 [[paper](https://arxiv.org/abs/2106.05731)]
- `CAVL`: Exploiting Class Activation Value for Partial-Label Learning. ICLR 2022 [[paper](https://openreview.net/pdf?id=qqdXHUGec9h)]
- `CRDPLL`: Revisiting Consistency Regularization for Deep Partial Label Learning. ICML 2022 [[paper](https://palm.seu.edu.cn/zhangml/files/ICML'22a.pdf)]
- `PiCO`: PICO: Contrastive Label Disambiguation for Partial Label Learning. ICLR2022 [[paper](https://arxiv.org/pdf/2007.08929)]
- `ABS-MAE ABS-GCE`: On the Robustness of Average Losses for Partial-Label Learning. TPAMI 2023 [[paper](https://openreview.net/pdf?id=qqdXHUGec9h)]

**LT-PLL**

- `Solar`: SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning. NeurIPS 2022 [[paper](https://arxiv.org/abs/2209.10365)]
- `RECORDS`: Long-Tailed Partial Label Learning via Dynamic Rebalancing. ICLR 2023 [[paper](https://arxiv.org/abs/2302.05080)]
- `HTC`: Long-tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation. AAAI 2024 [[paper](https://palm.seu.edu.cn/zhangml/files/AAAI'24c.pdf)]

**IDPLL**

- `ABLE`: Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning. IJCAI 2022 [[paper](https://www.ijcai.org/proceedings/2022/0502.pdf)]
- `IDGP`: Decompositional Generation Process for Instance-Dependent Partial Label Learning. ICLR 2023 [[paper](https://arxiv.org/abs/2204.03845)]
- `POP`: Progressive Purification for Instance-Dependent Partial Label Learning. ICML 2023 [[paper](https://arxiv.org/abs/2206.00830)]

### 🔎 Datasets

## Running on PLL Datasets

### Prepare the Dataset

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):

CIFAR-10 and CIFAR-100 datasets are widely used in the field of computer vision for image classification tasks. The following is a detailed introduction in Markdown:

### CIFAR-10 Dataset
- **Link**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### CIFAR-100 Dataset
- **Link**: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

These datasets play an important role in academic research, teaching experiments, and model training, and are very helpful for researchers and developers to study and evaluate image classification algorithms and models.

- CIFAR Dataset

```
Path/To/Dataset
├─ cifar-10-batches-py
│  ├─ batches.meta
│  ├─ data_batch_1
│  ├─ data_batch_2
│  ├─ data_batch_3
│  ├─ data_batch_4
│  ├─ data_batch_5
│  └─ test_batch
├─ cifar-100-python
│  ├─ meta
│  ├─ test
│  └─ train
└─ README.txt
```


## Running on Large-scale Long-tailed PLL Datasets

### Prepare the Dataset

Download the dataset [Places](http://places2.csail.mit.edu/download.html), [ImageNet](http://image-net.org/index), and [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018).

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):

- Places

```
Path/To/Dataset
├─ train
│  ├─ airfield
|  |  ├─ 00000001.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ airfield
   |  ├─ Places365_val_00000435.jpg
   |  └─ ......
   └─ ......
```

- ImageNet

```
Path/To/Dataset
├─ train
│  ├─ n01440764
|  |  ├─ n01440764_18.JPEG
|  |  └─ ......
│  └─ ......
└─ val
   ├─ n01440764
   |  ├─ ILSVRC2012_val_00000293.JPEG
   |  └─ ......
   └─ ......
```

- iNaturalist 2018

```
Path/To/Dataset
└─ train_val2018
   ├─ Actinopterygii
   |  ├─ 2229
   |  |  ├─ 2c5596da5091695e44b5604c2a53c477.jpg
   |  |  └─ ......
   |  └─ ......
   └─ ......
```

## Running on Instance-dependet PLL Datasets 
 ### Prepare the Dataset 
 Download the dataset [Stanford Dogs120](http://vision.stanford.edu/aditya86/ImageNetDogs/), [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Stanford Cars196] (can be downloaded from torchvision.datasets) and [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/). 
 Put files in the following locations and change the path in the data configure files in [configs/data](configs/data): 
 
 - Stanford Dogs120
```
Path/To/Stanford Dogs120
├─ train
│  ├─ Afghan_hound
|  |  ├─ n02088094_10074.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ Afghan_hound
   |  ├─ n02088094_993.jpg
   |  └─ ......
   └─ ......
```

- Stanford Dogs120
```
Path/To/Caltech-UCSD Birds-200-2011 (CUB-200-2011)
├─ train
│  ├─ 001.Black_footed_Albatross
|  |  ├─ Black_Footed_Albatross_0001_796111.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ 001.Black_footed_Albatross
   |  ├─ Black_Footed_Albatross_0046_56046.jpg
   |  └─ ......
   └─ ......
```


- Stanford Cars196
```
Path/To/Stanford Cars196
├─ train
│  ├─ 00001_am_1970_ford_mustang_gt.jpg
|  |  ├─ 00001_am_1970_ford_mustang_gt_0001.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ 00001_am_1970_ford_mustang_gt.jpg
   |  ├─ 00001_am_1970_ford_mustang_gt_0043.jpg
   |  └─ ......
   └─ ......
```


- FGVC-Aircraft
```
Path/To/FGVC-Aircraft
├─ train
│  ├─ 00001_airbus_a300.jpg
|  |  ├─ 00001_airbus_a300_0001.jpg
|  |  └─ ......
│  └─ ......
└─ val
   ├─ 00001_airbus_a300.jpg
   |  ├─ 00001_airbus_a300_0035.jpg
   |  └─ ......
   └─ ......
```


## ☄️ how to use

### 🕹️ Clone

Clone this GitHub repository:

```
git clone https://github.com/SEU-hk/UPB-A-Unified-Partial-label-learning-Benchmark.git
cd UPB-A-Unified-Partial-label-learning-Benchmark
```

### 🗂️ Dependencies

* Python 3.8
* PyTorch 2.0
* Torchvision 0.15
* Tensorboard

- Other dependencies are listed in [requirements.txt](requirements.txt).

To install requirements, run:

```sh
conda create -n lift python=3.8 -y
conda activate lift
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard
pip install -r requirements.txt
```

We encourage installing the latest dependencies. If there are any incompatibilities, please install the dependencies with the following versions.

```
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.2.1
yacs==0.1.8
tqdm==4.64.1
ftfy==6.1.1
regex==2022.7.9
timm==0.6.12
```



## 🔑 Quick Start

```bash
# PLL: run LIFT on CIFAR-100 (with partial_rate=0.1)  
python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l CC adaptformer True  

# LT-PLL: run LIFT on CIFAR-100-LT (with imbalanced ratio=100 and partial_rate=0.1)  
python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l HTC adaptformer True  

# IDPLL: run LIFT on fgvc100 (with pretrained wrn)   
python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP adaptformer True    
```

### Add new algoritms
```python
class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a partial-label learning algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
   # initialize an algorithm, including model, hparams, num_data, num_classes
    def __init__(self, model, input_shape, train_givenY, hparams):
        super(Algorithm, self).__init__()
        self.network = model
        self.hparams = hparams
        self.num_data = input_shape[0]
        self.num_classes = train_givenY.shape[1]

   # update step per minibatch
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step
        """
        raise NotImplementedError

   # model prediction
    def predict(self, x):
        raise NotImplementedError


### Add new datasets
1. Add yaml file in data dir  
- **dataset**: "CIFAR100_IR50"
- **root**: "./data"

2. Add dataloaders for new dataset


### Already included algorithms & datasets
|Type|Algorithms|Datasets|
|---|---|---|
|PLL|CC LWS CAVL CORR PRODEN PiCO ABS-MAE ABS-GCE|CIFAR-10 / CIFAR-100 |
|LT-PLL|Solar RECORDS HTC|CIFAR-10-LT / CIFAR-100-LT / Places-LT / ImageNet-LT |
|IDPLL|VALEN ABLE POP IDGP DIRK CEL|CIFAR-10 / CIFAR-100 / FGVC100 / CUB200 / Stanford Cars196 / Stanford DOGS120|

**PLL**: A instance corresponding to a candidate label set rather than a single label. 

**LT-PLL**: The number of instances follows a long-tailed distribution.

**IDPLL**: The noisy labels are very similar to the ground-truth label. The genneration process of candidate sets are dependent on instance itself.


### Hardware

Most experiments can be reproduced using a single GPU with 48GB of memory (larger models such as ViT-L require more memory).

- To further reduce the GPU memory cost, gradient accumulation is recommended. Please refer to [Usage](#usage) for detailed instructions.

### Reproduction

To reproduce the main result in the paper, please run

```bash
# run LIFT on ImageNet-LT
python main.py -d imagenet_lt -m clip_vit_b16 -p 0.1 -l Solar daptformer True

# run LIFT on Places-LT
python main.py -d places_lt -m clip_vit_b16 -p 0.1 -l Solar adaptformer True

# run LIFT on iNaturalist 2018
python main.py -d inat2018 -m clip_vit_b16_peft -p 0.1 -l Solar adaptformer True num_epochs 20
```

For other experiments, please refer to [scripts](scripts) for reproduction commands.

### Detailed Usage

To train and test the proposed method on more settings, run

```bash
python main.py -d [data] -m [model] -p [partial_rate] -l [loss_type] [options]
```

The `[data]` can be the name of a .yaml file in [configs/data](configs/data), including `imagenet_lt`, `places_lt`, `inat2018`, `cifar100_ir100`, `cifar100_ir50`, `cifar100_ir10`, etc.

The `[model]` can be the name of a .yaml file in [configs/model](configs/model), including `clip_rn50`, `clip_vit_b16`, `in21k_vit_b16`, etc.

The `[partial_rate]` refers to unifrom sampling strategy(uss) when 0 < p < 1; flip probability strategy (fps) when p = 0; instance dependent gengeration when p equal other values.

The `[loss_type]` can be any algorithm in file "algorithms.py"

Note that using only `-d` and `-m` options denotes only fine-tuning the classifier. Please use additional `[options]` for more settings. 

- To apply lightweight fine-tuning methods, add options like `lora True`, `adaptformer True`, etc.

- To apply test-time ensembling, add `tte True`.

Moreover, `[options]` can facilitate modifying the configure options in [utils/config.py](utils/config.py). Following are some examples.

- To specify the root path of datasets, add `root Path/To/Datasets`.

- To change the output directory, add an option like `output_dir NewExpDir`. Then the results will be saved in `output/NewExpDir`.

- To assign a single GPU (for example, GPU 0), add an option like `gpu 0`.

- To apply gradient accumulation, add `micro_batch_size XX`. This can further reduce GPU memory costs. Note that `XX` should be a divisor of `batch_size`.

- To test an existing model, add `test_only True`. This option will test the model trained by your configure file. To test another model, add an additional option like `model_dir output/AnotherExpDir`.

- To test an existing model on the training set, add `test_train True`.

You can also refer to [scripts](scripts) for example commands.

## 👨‍🏫 Acknowledgment

We thank the authors for the following repositories for code reference:


## Citation

If you find this repo useful for your work, please cite as:

