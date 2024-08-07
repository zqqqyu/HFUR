# HFUR
This is the official repository of "Hierarchical Frequency-based Upsampling and Refining for HEVC Compressed Video Enhancement". This repository contains *codes*, *video demos* of our work.

## Table of Content
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [Datasets](#datasets)


## Video Demos
Enhanced videos under various compression ratios. 
The videos have been compressed. Therefore, the results are inferior to that of the actual outputs.

QP37：

https://github.com/user-attachments/assets/923bcd93-c802-4293-bc48-f923d1da4cff

CBR200：

https://github.com/user-attachments/assets/88009ffa-5084-4c7a-9b7b-7722ffc135cc

https://github.com/user-attachments/assets/b0356363-51f1-49e6-b3b6-b632320bdacc

https://github.com/user-attachments/assets/2ae577a3-dc64-4923-8c54-5caebb7724e2

## Code
### Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Single GPU Training

> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/train_hfur.yml

### Single GPU Testing

> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/test_HFUR.yml

## Datasets
The directories used in the project need to be created manually. <br/>
Download LDV 1.0 dataset: [https://github.com/RenYang-home/NTIRE21_VEnh]
Hm16.20 is used to compress the standard test sequence. https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/HM-16.20 or https://data.vision.ee.ethz.ch/reyang/HM16.20.zip  <br/>
