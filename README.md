# Causal Mode Multiplexer
[CVPR 2024] Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection

[Taeheon Kim*](https://scholar.google.com/citations?user=9nYafYMAAAAJ&hl=en), [Sebin Shin*](https://scholar.google.com/citations?user=a-wpcQEAAAAJ&hl=en), [Youngjoon Yu](https://dblp.org/pid/266/1289.html), [Hak Gu Kim](https://scholar.google.com/citations?user=Jgh1JYgAAAAJ&hl=en), and [Yong Man Ro](https://scholar.google.com/citations?user=IPzfF7cAAAAJ&hl=en) (*: equally contributed)

<a href='https://arxiv.org/abs/2403.01300'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This repository contains code and links to the Causal Mode Multiplexer for unbiased multispectral pedestrian detection. We show that the Causal Mode Multiplexer framework effectively learns the causal relationships between multispectral inputs and predictions, thereby showing strong generalization ability on out-of-distribution data.



## Introduction

![Fig4_1](https://github.com/ssbin0914/Causal-Mode-Multiplexer/assets/101541087/7b90c4a0-ed92-464b-9bfb-9febe8f2d337)

We propose a novel Causal Mode Multiplexer (CMM) framework that performs unbiased inference from statistically biased multispectral pedestrian training data. Specifically, the CMM framework learns causality based on different cause-and-effects between ROTO<sup>1</sup>, RXTO, and ROTX inputs and predictions. For ROTO data, we guide the model to learn the total effect in the common mode learning scheme. Next, for ROTX and RXTO data, we utilize the tools of counterfactual intervention to eliminate the direct effect of thermal by subtracting it from the total effect. To this end, we modify the training objective from maximizing the posterior probability likelihood to maximizing the total indirect effect in the differential mode learning scheme. Our design requires combining two different learning schemes; therefore, we propose a Causal Mode Multiplexing (CMM) Loss to optimize the interchange.

##### <sup>1</sup>R⋆T⋆ refers to the visibility (O/X) in each modality. Generally, ROTO refers to daytime images, and RXTO refers to nighttime images. ROTX refers to daytime images in obscured situations.

## Installation

The following are the instructions on how to install dependencies.

First, clone the repository locally:

```bash
git clone https://github.com/ssbin0914/Causal-Mode-Multiplexer.git
```

Create conda env using the exported file 'pytorch.yaml' and then activate it:

```bash
conda env create -f pytorch.yaml
conda activate pytorch
```

## Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet50_lr0.007_Uncer_KL --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res50 --UKLoss ON
```

## Test
```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet50_lr0.007_Uncer_KL --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types all --UKLoss ON --net res50
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bash
@article{kim2024causal,
  title={Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection},
  author={Kim, Taeheon and Shin, Sebin and Yu, Youngjoon and Kim, Hak Gu and Ro, Yong Man},
  journal={arXiv preprint arXiv:2403.01300},
  year={2024}
}
```

## Acknowledgement

We thank the authors of the following research works and open-source projects.

[Uncertainty-Guided Cross-Modal Learning for Robust Multispectral Pedestrian Detection](https://ieeexplore.ieee.org/abstract/document/9419080?casa_token=2iNnZoAqg20AAAAA:lAH7D-i2BnLKOY8ZnLuK_fU-M2sZBg-nlQn5sUgw9ksBPpLVkqlCdCW3EfJ50N9-AHkAHt_J)

[Counterfactual VQA: A Cause-Effect Look at Language Bias](https://github.com/yuleiniu/cfvqa?tab=readme-ov-file)
