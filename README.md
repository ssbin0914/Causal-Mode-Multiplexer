# Causal Mode Multiplexer
[CVPR 2024] Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection

[Taeheon Kim*](https://scholar.google.com/citations?user=9nYafYMAAAAJ&hl=en), [Sebin Shin*](https://scholar.google.com/citations?user=a-wpcQEAAAAJ&hl=en), [Youngjoon Yu](https://dblp.org/pid/266/1289.html), [Hak Gu Kim](https://scholar.google.com/citations?user=Jgh1JYgAAAAJ&hl=en), and [Yong Man Ro](https://scholar.google.com/citations?user=IPzfF7cAAAAJ&hl=en) (*: equally contributed)

<a href='https://arxiv.org/abs/2403.01300'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This repository contains code and links to the Causal Mode Multiplexer for unbiased multispectral pedestrian detection. We show that the Causal Mode Multiplexer framework effectively learns the causal relationships between multispectral inputs and predictions, thereby showing strong generalization ability on out-of-distribution data.



## Introduction

![Fig4_1](https://github.com/ssbin0914/Causal-Mode-Multiplexer/assets/101541087/7b90c4a0-ed92-464b-9bfb-9febe8f2d337)

We propose a novel Causal Mode Multiplexer (CMM) framework that performs unbiased inference from statistically biased multispectral pedestrian training data. Specifically, the CMM framework learns causality based on different cause-and-effects between ROTO<sup>1</sup>, RXTO, and ROTX inputs and predictions. For ROTO data, we guide the model to learn the total effect in the common mode learning scheme. Next, for ROTX and RXTO data, we utilize the tools of counterfactual intervention to eliminate the direct effect of thermal by subtracting it from the total effect. To this end, we modify the training objective from maximizing the posterior probability likelihood to maximizing the total indirect effect in the differential mode learning scheme. Our design requires combining two different learning schemes; therefore, we propose a Causal Mode Multiplexing (CMM) Loss to optimize the interchange.

##### <sup>1</sup>R⋆T⋆ refers to the visibility (O/X) in each modality. Generally, ROTO refers to daytime images, and RXTO refers to nighttime images. ROTX refers to daytime images in obscured situations.

## Code will be released soon!
