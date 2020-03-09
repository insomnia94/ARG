# PyTorch Implementation of ARG

## Introduction

This repository is Pytorch implementation of [Adaptive ROI Generation for Video Object Segmentation Using Reinforcement Learning]
Check our [paper](https://arxiv.org/pdf/1909.12482.pdf) for more details.

## Prerequisites

* Python 3.5
* Pytorch 0.4.1
* CUDA 8.0

## Installation

Please refer to [OnAVOS](https://github.com/lichengunc/MAttNet) to install [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn), [REFER](https://github.com/lichengunc/refer) and [refer-parser2](https://github.com/lichengunc/refer-parser2).
Follow Step 1 & 2 in Training to prepare the data and features.

## Training

Train ARG with ground-truth annotation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py --dataset ${DATASET} --splitBy ${SPLITBY} --exp_id ${EXP_ID}
```

## Evaluation

Evaluate ARG with ground-truth annotation:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval.py --dataset ${DATASET} --splitBy ${SPLITBY} --split ${SPLIT} --id ${EXP_ID}
```


## Citation

    @article{sun2019adaptive,
    title={Adaptive ROI Generation for Video Object Segmentation Using Reinforcement Learning},
    author={Sun, Mingjie and Xiao, Jimin and Lim, Eng Gee and Xie, Yanchu and Feng, Jiashi},
    journal={arXiv preprint arXiv:1909.12482},
    year={2019}
    }


## Acknowledgement

Thanks for the work of [Licheng Yu](http://cs.unc.edu/~licheng/). Our code is based on the implementation of [OnAVOS](https://github.com/lichengunc/MAttNet).
