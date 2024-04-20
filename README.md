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

The following are the instructions on how to install dependencies and prepare data. The code is tested on `torch=0.3.1, cuda9.0`.

Step 1. Clone the repository locally:

```bash
git clone https://github.com/ssbin0914/Causal-Mode-Multiplexer.git
cd Causal-Mode-Multiplexer
```

Step 2. Create conda env using the file `requirements.txt` and then activate it:

```bash
conda create -n cmm python=2.7.16
conda activate cmm
pip install -r requirements.txt
wget https://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision==0.1.8
cd lib
sh make.sh
cd ..
```

Step 3. Download the `data` folder from this [link](https://drive.google.com/file/d/1wgZtVGwJW-02XKSonyz_nKKwoNlu86vm/view?usp=sharing) and put it under the `Causal Mode Multiplexer/` directory. We provide the FLIR dataset.

```
└── Causal-Mode-Multiplexer
               ├── cfgs
               │
               ├── lib
               │ 
               ├── data
               │    ├── cache
               │    │ 
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
               │    │ 
               │    └── pretrained_model
               │            ├── resnet50.pth
               │            ├── resnet101.pth
               │            └── ...
               │
               └── ...
```

## 🔨Training

To train the CMM, simply run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet50_lr0.007_Uncer_KL --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res50 --UKLoss ON --lr 0.007 --lr_decay_step 1 --epochs 2
```

where `ResNet50_lr0.007_Uncer_KL` is the name of the folder where the weights will be stored. `--lr` specifies the learning rate, `--lr_decay_step` indicates the step at which the learning rate decays, and `--epochs` refers to the number of training epochs.

After running the code, the weights are stored in `weights/res50/kaist/ResNet50_lr0.007_Uncer_KL/` directory.

* The pretrained weight for the FLIR dataset are available from this [link](https://drive.google.com/file/d/1-zwQI536o65FEfzoaQFU7hLurW1z4bWt/view?usp=sharing). If you want to test with this pretrained weight, put the weight file under the `weights/res50/kaist/ResNet50_lr0.007_Uncer_KL/` directory.

## 🧪Test

To test the CMM, simply run:

```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet50_lr0.007_Uncer_KL --dataset kaist --cuda --cag --checksession 2 --checkepoch 2 --checkpoint 1381 --types all --UKLoss ON --net res50 --vis
```

where `ResNet50_lr0.007_Uncer_KL` is the name of the folder containing the weights you want to test. Set `--checksession`, `--checkepoch`, and `--checkpoint` according to the name of your weight file, e.g., `fpn_2_2_1381.pth`.

After running the code, the detection results are stored in `Detection_Result/` directory. These results are used for evaluation purposes. Visualization results are also stored as RGB images in the `images` folder and as infrared (IR) images in the `images_ir` folder.

To calculate the AP score, we use MATLAB.<br>
Step 1. Create a folder locally and then create a `test` folder inside it.<br>
Step 2. Move the txt files from the `Detection_Result/` directory into the test folder.<br>
Step 3. Download and unzip the ground truth annotation folder from this [link](https://drive.google.com/file/d/1mhzmFKpvzjK9P1UzYc1btaJgezWxEaFk/view?usp=sharing).<br>
Step 4. Download the evaluator from this [link](https://drive.google.com/drive/folders/1XL_208QF2QEqOQ9isM_riZY9bGb-B--i?usp=sharing).<br>
Step 5. Open `FLIRdevkit-matlab-wrapper/demo_test.m`. In this file, set `dtDir` to the path of the test folder and `gtDIR` to the path of the downloaded ground truth annotation folder.<br>
Step 6. Open `bbGt.m` and set a breakpoint at line 761 in `bbGt.m`. Then run `demo_test.m`. When it hits the breakpoint, enter `trapz(xs, ys)`. This value is the AP score.

## ✨New Dataset: ROTX-MP

To evaluate modality bias in multispectral pedestrian detectors, we propose a new dataset: ROTX Multispectral Pedestrian (ROTX-MP) dataset. It mainly contains ROTX data, compared to existing datasets that consist of ROTO and RXTO data. ROTX-MP consists of 1000 ROTX test images collected from two practical scenarios (pedestrians over a glass window, pedestrians wearing heat-insulation clothes) related to the applications of multispectral pedestrian detection.

If you need the ROTX-MP dataset, feel free to email eetaekim@kaist.ac.kr.

To evaluate performance on the ROTX-MP dataset:<br>
Step 1. Place the ground truth annotations in the `lwir` and `visible` folders within the `data/KAIST_PED/Annotations/` directory.<br>
Step 2. Put the images from the ROTX-MP dataset into the `lwir` and `visible` folders located in the `data/KAIST_PED/JPEG_Images/` directory.<br>
Step 3. Replace the `test.txt` file in the `data/KAIST_PED/ImageSets/Main/` directory with the ROTX-MP `test.txt` file. Note that the original `test.txt` file is from the FLIR dataset.<br>
Step 4. Delete the files located in the `data/cache/` directory if you evaluate on the ROTX-MP dataset after evaluating the FLIR dataset. It is crucial to remove these files when switching between datasets for training and evaluation. This is also the case when you want to evaluate on the FLIR dataset after evaluating the ROTX-MP dataset.

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
