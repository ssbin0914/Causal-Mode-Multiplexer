## [CVPR 2024] Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection
Authors: [Taeheon Kim*](https://scholar.google.com/citations?user=9nYafYMAAAAJ&hl=en), [Sebin Shin*](https://scholar.google.com/citations?user=a-wpcQEAAAAJ&hl=en), Youngjoon Yu, [Hak Gu Kim](https://scholar.google.com/citations?user=Jgh1JYgAAAAJ&hl=en), and [Yong Man Ro](https://scholar.google.com/citations?user=IPzfF7cAAAAJ&hl=en) (*: equally contributed)

Official PyTorch implementation of the paper "[Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection](https://arxiv.org/abs/2403.01300)"

### Introduction
![Fig4_1](https://github.com/ssbin0914/Causal-Mode-Multiplexer/assets/101541087/7b90c4a0-ed92-464b-9bfb-9febe8f2d337)

We propose a novel Causal Mode Multiplexer (CMM) framework that performs unbiased inference from statistically biased multispectral pedestrian training data. Specifically, the CMM framework learns causality based on different cause-and-effects between ROTO, RXTO, and ROTX inputs and predictions. For ROTO data, we guide the model to learn the total effect in the common mode learning scheme. Next, for ROTX and RXTO data, we utilize the tools of counterfactual intervention to eliminate the direct effect of thermal by subtracting it from the total effect. To this end, we modify the training objective from maximizing the posterior probability likelihood to maximizing the total indirect effect in the differential mode learning scheme. Our design requires combining two different learning schemes; therefore, we propose a Causal Mode Multiplexing (CMM) Loss to optimize the interchange.
