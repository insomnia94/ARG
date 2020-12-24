## Prerequisites

* Python 3.6
* Tensorflow 1.3.0
* CUDA 8.0

## Installation

Please refer to [OnAVOS](https://github.com/Stocastico/OnAVOS) to prepare related dataset and install the environment of the benchmark method.

## Training

Train ARG on DAVIS 2016:

```bash
python ./train.py
```

## Evaluation

Evaluate ARG on DAVIS 2016:

```bash
python ./eval.py configs/DAVIS16_online
```


## Acknowledgement

Thanks for the work of [Voigtlaender Paul](https://www.vision.rwth-aachen.de/person/197/). Our code is based on the implementation of [OnAVOS](https://github.com/Stocastico/OnAVOS).
