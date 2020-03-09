# Tensorflow Implementation of ARG

## Introduction

This repository is Tensorflow implementation of [Adaptive ROI Generation for Video Object Segmentation Using Reinforcement Learning]
Check our [paper](https://arxiv.org/pdf/1909.12482.pdf) for more details.

## Prerequisites

* Python 3.6
* Tensorflow 1.3.0
* CUDA 8.0

## Installation

Please refer to [OnAVOS](https://github.com/Stocastico/OnAVOS) to install the environment of the benchmark method.

## Training

Train ARG with ground-truth annotation:

```bash
python ./train.py
```

## Evaluation

Evaluate ARG with ground-truth annotation:

```bash
python ./eval.py configs/DAVIS16_online
```


## Citation

    @article{sun2019adaptive,
    title={Adaptive ROI Generation for Video Object Segmentation Using Reinforcement Learning},
    author={Sun, Mingjie and Xiao, Jimin and Lim, Eng Gee and Xie, Yanchu and Feng, Jiashi},
    journal={arXiv preprint arXiv:1909.12482},
    year={2019}
    }


## Acknowledgement

Thanks for the work of [Voigtlaender Paul](https://www.vision.rwth-aachen.de/person/197/). Our code is based on the implementation of [OnAVOS](https://github.com/Stocastico/OnAVOS).
