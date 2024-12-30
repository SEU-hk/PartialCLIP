# UPB: A-Unified-Partial-label-learning-Benchmark

This is the source code for the paper.

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
1 Add yaml file in data dir  
# Dataset and Root Configuration
dataset: "CIFAR100_IR50"
root: "./data"

2 Add dataloaders for new dataset


## Already included algorithms & datasets
|Type|Algorithms|Datasets|
|---|---|---|
|PLL|CC LWS CAVL CORR PRODEN PiCO ABS-MAE ABS-GCE|Cifar10 Cifar100|
|LT-PLL|Solar RECORDS HTC|Cifar100 Places-LT ImageNet-LT|
|IDPLL|VALEN ABLE POP IDGP DIRK CEL|Cifar10 FGVC100 Cub200|

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

## Running on Real-world PLL Datasets

### Prepare the Dataset

Download the dataset [Tabular](https://palm.seu.edu.cn/zhangml/).

Put files in the following locations and change the path in the data configure files in [configs/data](configs/data):

- **FG-NET data**:
    - **Description**: Facial age estimation from crowd-sourced annotations.
    - **Reference**: G. Panis, A. Lanitis. An overview of research activities in facial age estimation using the FG-NET aging database. Lecture Notes in Computer Science 8926, Berlin: Springer, 2015, 737-750.
    - **Size**: 1.98Mb
- **Lost data**:
    - **Description**: Automatic face naming from videos.
    - **Reference**: T. Cour, B. Sapp, B. Taskar. Learning from partial labels. Journal of Machine Learning Research, 12(May): 1501–1536, 2011.
    - **Size**: 914Kb
- **MSRCv2 data**:
    - **Description**: Object classification.
    - **Reference**: L. Liu, T. Dietterich. A conditional multinomial mixture model for superset label learning. In: Advances in Neural Information Processing Systems 25, Cambridge, MA: MIT Press, 2012, 557–565.
    - **Size**: 373Kb
- **BirdSong data**:
    - **Description**: Bird song classification.
    - **Reference**: F. Briggs, X. Z. Fern, R. Raich. Rank-loss support instance machines for MIML instance annotation. In: Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Beijing, China, 2012, 534–542.
    - **Size**: 1.00Mb
- **Soccer Player data**:
    - **Description**: Automatic face naming from images.
    - **Reference**: Z. Zeng, S. Xiao, K. Jia, T.-H. Chan, S. Gao, D. Xu, Y. Ma. Learning by associating ambiguously labeled images. In: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, Portland, OR, 2013, 708–715.
    - **Size**: 35.18Mb
- **Yahoo! News**:
    - **Description**: Automatic face naming from images.
    - **Reference**: M. Guillaumin, J. Verbeek, C. Schmid. Multiple instance metric learning from automatically labeled bags of faces. In: Lecture Notes in Computer Science 6311, Berlin: Springer, 2010, 634–637.
    - **Size**: 28.04Mb
- **Mirflickr data**:
    - **Description**: Web image classification.
    - **Reference**: M. J. Huiskes, M. S. Lew. The MIR Flickr retrieval evaluation. In: Proceedings of the 1st ACM International Conference on Multimedia Information Retrieval, Vancouver, Canada, 2008, 39–43.
    - **Size**: 30.40Mb
 
# Example
- Lost

```
Path/To/Dataset
├─ data
├─ partial-target
└─ target

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
 
 - Places 


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
