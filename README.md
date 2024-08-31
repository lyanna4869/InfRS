# InfRS: 

Arxiv: https://arxiv.org/abs/2405.11293

## Contact

```
If you have any questions, please file an issue on this github repo.
```

## Installation

This project is modified based on the fsce (https://github.com/megvii-research/FSCE.git) project. 

FsDet is built on [Detectron2](https://github.com/facebookresearch/detectron2). But you don't need to build detectron2 seperately as this codebase is self-contained. You can follow the instructions below to install the dependencies and build `FsDet`. FSCE functionalities are implemented as `class`and `.py` scripts in FsDet which therefore requires no extra build efforts. 

**Dependencies**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9

**Build**

```bash
python setup.py build develop 
```


Note: you may need to rebuild FsDet after reinstalling a different build of PyTorch.


## Data preparation

We adopt the same benchmarks as in FsDet, including three datasets: NWPU VHR-10, DIOR, RSOD. All datasets are converted into the format of VOC2007. 

- NWPU VHR-10: We randomly split the 10 object classes into 7 base classes and 3 novel classes, and we consider 3 random splits. The splits can be found in [fs_nwpu/core/meta.py](fs_nwpu/core/meta.py).
- DIOR: We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fs_dior/core/meta.py](fs_dior/core/meta.py).
- RSOD: We randomly split the 4 object classes into 3 base classes and 1 novel classes, and we consider 3 random splits. The splits can be found in [fs_rsod/core/meta.py](fs_rsod/core/meta.py).

The datasets and data splits are built-in, simply make sure the directory structure agrees with [datasets/README.md](datasets/README.md) to launch the program. 

# Training Instructions

## Stage 1: Base Training

Related code for different dataset are located in `fs_xxxx` where xxxx can be replaced by `nwpu`, `dior` or `rsod`, 

First train a base model. To train a base model on the first split of PASCAL VOC, run
```bash
python fs_nwpu/train_fasterrcnn.py --config-file configs/NWPU/base_training/base1.yaml
```

## Stage 2: Prototype extraction

First, we need to collect the prototype for all base objects:

```bash
python fs_nwpu/prototype_fasterrcnn.py --config-file configs/NWPU/prototype/split1.yml
```

Now, we can convert these collected feature into prototype for these base categories.

```bash
python fs_nwpu/scripts/convert_feature_as_prototype.py
```

## Stage 3: Few-Shot Fine-Tuning

### Initialization

After training the base model, run ```fs_nwpu/scripts/ckpt_redetect.py``` to obtain an initialization for the full model. We only modify the weights of the last layer of the detector, while the rest of the network are kept the same. The weights corresponding to the base classes are set as those obtained in the previous stage, and the weights corresponding to the novel classes are either randomly initialized or set as those of a predictor fine-tuned on the novel set.

#### Novel Weights

To use novel weights, fine-tune a predictor on the novel set. We reuse the base model trained in the previous stage but retrain the last layer from scratch. First remove the last layer from the weights file by running
```bash
python fs_nwpu/scripts/ckpt_surgery.py --src1 weights/NWPU_R101_split1.pth --method randinit  --save-dir work_dirs/nwpu_resnet101_base1_all_redetect/
```


### Fine-Tuning for novel data.

Next, fine-tune the predictor on the novel set by running
```bash
python fs_nwpu/inc_train_fasterrcnn.py --config-file configs/NWPU/split1/inc/10_shot_INC_CLP.yaml 
```

#### Evaluation

To evaluate the trained models, run

```bash
python fs_nwpu/test_fasterrcnn.py --config-file configs/NWPU/split1/inc/10shot_INC_CLP.yml --eval-only
```



