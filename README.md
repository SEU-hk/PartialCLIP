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

Welcome to (**UPB**): a unified partial label learning benchmark for classification. The UPB framework proposed by us has a clear structure. It integrates the state-of-the-art (SOTA) algorithms of (**PLL 2024**) (Partial Label Learning), (**LT-PLL**) (Long-tailed Partial Label Learning), and (**IDPLL**)IDPLL (Instance-dependent Partial Label Learning), and provides a unified interface. Moreover, its code has been open-sourced on GitHub, allowing new methods and datasets to be added easily. 


## 📰 What's New
- [2024-12]🌟 Add [MOS](https://arxiv.org/abs/2412.09441). State-of-the-art method of 2025!
- [2024-12]🌟 Check out our [latest work](https://arxiv.org/abs/2412.09441) on pre-trained model-based class-incremental learning (**AAAI 2025**)!
- [2024-10]🌟 Check out our [latest work](https://arxiv.org/abs/2410.00911) on pre-trained model-based domain-incremental learning! 
- [2024-08]🌟 Check out our [latest work](https://arxiv.org/abs/2303.07338) on pre-trained model-based class-incremental learning (**IJCV 2024**)!
- [2024-07]🌟 Check out our [rigorous and unified survey](https://arxiv.org/abs/2302.03648) about class-incremental learning, which introduces some memory-agnostic measures with holistic evaluations from multiple aspects (**TPAMI 2024**)!
- [2024-07]🌟 Check out our [work about all-layer margin in class-incremental learning](https://openreview.net/forum?id=aksdU1KOpT) (**ICML 2024**)!
- [2024-04]🌟 Check out our [latest survey](https://arxiv.org/abs/2401.16386) on pre-trained model-based continual learning (**IJCAI 2024**)!
- [2024-03]🌟 Add [EASE](https://arxiv.org/abs/2403.12030). State-of-the-art method of 2024!
- [2024-03]🌟 Check out our [latest work](https://arxiv.org/abs/2403.12030) on pre-trained model-based class-incremental learning (**CVPR 2024**)!
- [2023-12]🌟 Add RanPAC.
- [2024-10]🌟 Add SoTa (*PLL*) baselines, including [CC](https://arxiv.org/pdf/2007.08929)(**NeurIPS 2020**), [LWS](https://arxiv.org/abs/2106.05731)(**ICML 2021**), [CAV](https://openreview.net/pdf?id=qqdXHUGec9h)(**ICLR 2022**), [PRODEN](https://arxiv.org/abs/2002.08053)(**ICML 2020**)!
- [2023-09]🌟 Initial version of UPB is released.
- [2024-09]🌟 Fistly, conduct experiments on [RECORDS](https://arxiv.org/abs/2302.05080)(*LT-PLL*)(**CVPR 2024**) with UPB!

## Requirements

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

## Add new algoritms
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

# Example
class CC(Algorithm):
    """
    CC
    Reference: Provably consistent partial-label learning, NeurIPS 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(CC, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        loss = self.cc_loss(self.predict(x), partial_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def cc_loss(self, outputs, partialY):
        sm_outputs = F.softmax(outputs, dim=1)
        final_outputs = sm_outputs * partialY
        average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
        return average_loss  

    def predict(self, x):
        return self.network(x)[0]

```

## Add new datasets
1. Add yaml file in data dir  
- **dataset**: "CIFAR100_IR50"
- **root**: "./data"

2. Add dataloaders for new dataset
```python
def load_dogs120(cfg, transform_train, transform_test):
    original_train = Dogs(root=cfg.root, train=True, cropped=False, transform=transform_train, download=True)
    original_full_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=len(original_train),
                                                       shuffle=False, num_workers=20)
    ori_data, ori_labels = next(iter(original_full_loader))
    ori_labels = ori_labels.long()
    
    # 计算数据集中样本的总数量
    num_instances = len(original_train)
    classnames = original_train.classes
 
    # 正确计算类别数量，通过获取所有标签的去重后的集合的长度来确定类别数
    num_classes = len(classnames)
    print(num_classes)

    test_dataset = Dogs(root=cfg.root, train=False, cropped=False, transform=transform_test, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8)

    model = models.wide_resnet50_2()
    model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
    model = model.cuda()
    model.load_state_dict(
        torch.load(os.path.expanduser('weights/dogs120.pt'))['model_state_dict'])
    partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels, 0.1)

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1

    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')

    partial_training_dataset = DOGS160_Partialize(
        [os.path.join(original_train.images_folder, tup[0]) for tup in original_train._flat_breed_images],
        [tup[1] for tup in original_train._flat_breed_images], partialY_matrix.float(), ori_labels.float())

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    return partial_training_dataloader, partialY_matrix, test_loader, num_instances, num_classes, classnames
```

## Already included algorithms & datasets
|Type|Algorithms|Datasets|
|---|---|---|
|PLL|CC LWS CAVL CORR PRODEN PiCO ABS-MAE ABS-GCE|CIFAR-10 / CIFAR-100 |
|LT-PLL|Solar RECORDS HTC|CIFAR-10-LT / CIFAR-100-LT / Places-LT / ImageNet-LT |
|IDPLL|VALEN ABLE POP IDGP DIRK CEL|CIFAR-10 / CIFAR-100 / FGVC100 / CUB200 / Stanford Cars196 / Stanford DOGS120|

PLL: A instance corresponding to a candidate label set rather than a single label. Two strategies, namely Uniform Sampling Strategy (USS) and Flip Probalbility Sampling Strategy (FPS)(Lv et al., 2020), which randomly generate candidate label sets in PLL, are adopted. 

LT-PLL: The number of instances follows a long-tailed distribution.

IDPLL: The noisy labels are very similar to the ground-truth label. The genneration process of candidate sets are dependent on instance itself.


## Hardware

Most experiments can be reproduced using a single GPU with 48GB of memory (larger models such as ViT-L require more memory).

- To further reduce the GPU memory cost, gradient accumulation is recommended. Please refer to [Usage](#usage) for detailed instructions.

## Quick Start

```bash
# PLL: run LIFT on CIFAR-100 (with partial_rate=0.1)  
python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l CC adaptformer True  

# LT-PLL: run LIFT on CIFAR-100-LT (with imbalanced ratio=100 and partial_rate=0.1)  
python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l HTC adaptformer True  

# IDPLL: run LIFT on fgvc100 (with pretrained wrn)   
python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP adaptformer True    
```

## Running on PLL Datasets

### Prepare the Dataset

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):

CIFAR-10 and CIFAR-100 datasets are widely used in the field of computer vision for image classification tasks. The following is a detailed introduction in Markdown:

### CIFAR-10 Dataset
- **Basic Information**:
    - It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The 10 classes include airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.
    - The dataset is divided into a training set of 50,000 images and a test set of 10,000 images.
- **Format**: In the Python version, the dataset is stored in pickle files. Each batch file contains a dictionary with two elements: "data" and "labels". "data" is a 10000x3072 numpy array of uint8s, where each row stores a 32x32 color image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. "labels" is a list of 10000 numbers in the range 0-9, indicating the label of each image. There is also a "batches.meta" file, which contains a 10-element list "label_names", giving meaningful names to the numeric labels.
- **Link**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### CIFAR-100 Dataset
- **Basic Information**:
    - It also contains 60,000 32x32 color images, but is divided into 100 classes, with 600 images per class. Additionally, the 100 classes in the CIFAR-100 are grouped into 20 super-classes, and each image is associated with a "fine" label (the class it is associated with) and a "coarse" label (the superclass it is associated with).
- **Format**: Similar to the CIFAR-10 dataset, it is also stored in pickle files in the Python version. Each batch file contains a dictionary with "data" and "labels" elements. "data" stores the image data in a specific format, and "labels" is a list of numbers indicating the class labels of the images.
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

## Acknowledgment

We thank the authors for the following repositories for code reference:


## Citation

If you find this repo useful for your work, please cite as:

```bibtex
@inproceedings{shi2024longtail,
  title={Long-Tail Learning with Foundation Model: Heavy Fine-Tuning Hurts},
  author={Jiang-Xin Shi and Tong Wei and Zhi Zhou and Jie-Jing Shao and Xin-Yan Han and Yu-Feng Li},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```
