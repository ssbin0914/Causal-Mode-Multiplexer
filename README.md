# Causal Mode Multiplexer
[CVPR 2024] Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection

<p>
  <a href="https://scholar.google.com/citations?user=9nYafYMAAAAJ&hl=en">Taeheon Kim</a><sup>*</sup>, 
  <a href="https://scholar.google.com/citations?user=a-wpcQEAAAAJ&hl=en">Sebin Shin</a><sup>*</sup>, 
  <a href="https://dblp.org/pid/266/1289.html">Youngjoon Yu</a>, 
  <a href="https://scholar.google.com/citations?user=Jgh1JYgAAAAJ&hl=en">Hak Gu Kim</a>, 
  and <a href="https://scholar.google.com/citations?user=IPzfF7cAAAAJ&hl=en">Yong Man Ro</a> 
  ( * : equally contributed)
</p>

<a href='https://arxiv.org/abs/2403.01300'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This repository contains code and links to the Causal Mode Multiplexer for unbiased multispectral pedestrian detection. We show that the Causal Mode Multiplexer framework effectively learns the causal relationships between multispectral inputs and predictions, thereby showing strong generalization ability on out-of-distribution data.

## 📢News

- **2024.04.17** Code released. (Update in progress)
- **2024.03.02** [**arXiv**](https://arxiv.org/abs/2403.01300) preprint released. 
- **2024.02.27** The paper of Causal Mode Multiplexer has been accepted to CVPR 2024.

## 📝Summary

![Fig4_1](https://github.com/ssbin0914/Causal-Mode-Multiplexer/assets/101541087/7b90c4a0-ed92-464b-9bfb-9febe8f2d337)

We propose a novel Causal Mode Multiplexer (CMM) framework that performs unbiased inference from statistically biased multispectral pedestrian training data. Specifically, the CMM framework learns causality based on different cause-and-effects between ROTO<sup>1</sup>, RXTO, and ROTX inputs and predictions. For ROTO data, we guide the model to learn the total effect in the common mode learning scheme. Next, for ROTX and RXTO data, we utilize the tools of counterfactual intervention to eliminate the direct effect of thermal by subtracting it from the total effect. To this end, we modify the training objective from maximizing the posterior probability likelihood to maximizing the total indirect effect in the differential mode learning scheme. Our design requires combining two different learning schemes; therefore, we propose a Causal Mode Multiplexing (CMM) Loss to optimize the interchange.

##### <sup>1</sup>R⋆T⋆ refers to the visibility (O/X) in each modality. Generally, ROTO refers to daytime images, and RXTO refers to nighttime images. ROTX refers to daytime images in obscured situations.

## 🔧Installation & Data Preparation

The following are the instructions on how to install dependencies.

Step 1. Clone the repository locally:

```bash
git clone https://github.com/ssbin0914/Causal-Mode-Multiplexer.git
```

Step 2. Create conda env using the exported file `pytorch.yaml` and then activate it:

```bash
conda env create -f pytorch.yaml
conda activate pytorch
```

Step 3. Download the `data` folder from [link](). We provide the FLIR dataset.

```
└── Causal-Mode-Multiplexer
               ├── cfgs
               ├── lib
               ├── data
               │    ├── cache
               │    ├── KAIST_PED
               │    │       └── Annotations
               │    │       │        ├── lwir
               │    │       │        │     ├── FLIR_08864.txt
               │    │       │        │     └── ...
               │    │       │        └── visible
               │    │       │              ├── FLIR_08864.txt
               │    │       │              └── ...
               │    │       ├── annotations_cache
               │    │       ├── ImageSets
               │    │       │        ├── Main 
               │    │       │        │     ├── train.txt
               │    │       │        │     └── test.txt
               │    │       │        └── Main_Org
               │    │       ├── JPEGImages
               │    │       │        ├── lwir
               │    │       │        │     ├── FLIR_08864.jpg
               │    │       │        │     └── ...
               │    │       │        └── visible
               │    │       │              ├── FLIR_08864.jpg
               │    │       │              └── ...
               │    │       └── results
               │    └── pretrained_model
               │            ├── resnet50.pth
               │            ├── resnet101.pth
               │            └── ...
               │
               │
               ├── Detection_Result
               ├── export
               ├── images
               ├── images_ir
               ├── output
               │    └── res50
               │            └── kaist_test
               │                     └── faster_rcnn_10
               ├── weights
               │    └── res50
               │            └── kaist
               └── ...
```

## 🔨Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet50_lr0.007_Uncer_KL --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res50 --UKLoss ON --lr 0.007 --lr_decay_step 1 --epochs 2
```

## 🧪Test
```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet50_lr0.007_Uncer_KL --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 1381 --checksession 2 --types all --UKLoss ON --net res50 --vis
```

## ✨New Dataset: ROTX-MP

To evaluate modality bias in multispectral pedestrian detectors, we propose a new dataset: ROTX Multispectral Pedestrian (ROTX-MP) dataset. It mainly contains ROTX data, compared to existing datasets that consist of ROTO and RXTO data. ROTX-MP consists of 1000 ROTX test images collected from two practical scenarios (pedestrians over a glass window, pedestrians wearing heat-insulation clothes) related to the applications of multispectral pedestrian detection.

If you need the ROTX-MP dataset, feel free to email eetaekim@kaist.ac.kr. 

## 📃Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{kim2024causal,
  title={Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection},
  author={Kim, Taeheon and Shin, Sebin and Yu, Youngjoon and Kim, Hak Gu and Ro, Yong Man},
  journal={arXiv preprint arXiv:2403.01300},
  year={2024}
}
```

## 🙏Acknowledgement

We thank the authors of the following research works and open-source projects. We've used some of the code from different repositories.

[Uncertainty-Guided Cross-Modal Learning for Robust Multispectral Pedestrian Detection](https://ieeexplore.ieee.org/abstract/document/9419080?casa_token=2iNnZoAqg20AAAAA:lAH7D-i2BnLKOY8ZnLuK_fU-M2sZBg-nlQn5sUgw9ksBPpLVkqlCdCW3EfJ50N9-AHkAHt_J)

[Counterfactual VQA: A Cause-Effect Look at Language Bias](https://github.com/yuleiniu/cfvqa?tab=readme-ov-file)
