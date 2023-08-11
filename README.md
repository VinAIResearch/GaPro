##### Table of contents
1. [Installation](#Installation)
2. [Data Preparation](#Data-Preparation)
3. [Training and Testing](#Training-and-Testing) 
4. [Quick Demo](#Quick-Demo)
5. [Visualization](#Qualitative-Results)
6. [Acknowledgments](#Acknowledgments)
7. [Contacts](#Contacts)

# GaPro: Box-Supervised 3D Point Cloud Instance Segmentation \\Using Gaussian Processes as Pseudo Labelers

<a href="https://arxiv.org/abs/2307.13251"><img src="https://img.shields.io/badge/https%3A%2F%2Farxiv.org%2Fabs%2F2307.13251-arxiv-brightgreen"></a>

[Tuan Duc Ngo](https://ngoductuanlhp.github.io/),
[Binh-Son Hua](https://sonhua.github.io/),
[Khoi Nguyen](https://www.khoinguyen.org/)<br>
VinAI Research, Vietnam

> **Abstract**: 
Instance segmentation on 3D point clouds (3DIS) is a longstanding challenge in computer vision, where state-of-the-art methods are mainly based on full supervision. As annotating ground truth dense instance masks is tedious and expensive, solving 3DIS with weak supervision has become more practical. In this paper, we propose GaPro, a new instance segmentation for 3D point clouds using axis-aligned 3D bounding box supervision. Our two-step approach involves generating pseudo labels from box annotations and training a 3DIS network with the resulting labels. Additionally, we employ the self-training strategy to improve the performance of our method further. We devise an effective Gaussian Process to generate pseudo instance masks from the bounding boxes and resolve ambiguities when they overlap, resulting in pseudo instance masks with their uncertainty values. Our experiments show that GaPro outperforms previous weakly supervised 3D instance segmentation methods and has competitive performance compared to state-of-the-art fully supervised ones. Furthermore, we demonstrate the robustness of our approach, where we can adapt various state-of-the-art fully supervised methods to the weak supervision task by using our pseudo labels for training.
![overview](docs/gapro_arch.png)

Details of the model architecture and experimental results can be found in [our paper](https://arxiv.org/abs/2307.13251):

```bibtext
@inproceedings{ngo2023gapro,
 author={Tuan Duc Ngo, Binh-Son Hua, Khoi Nguyen},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
 title={GaPro: Box-Supervised 3D Point Cloud Instance Segmentation \\Using Gaussian Processes as Pseudo Labelers},
 year= {2023}
}
```

**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

## Installation :memo:
Please refer to [installation guide](docs/INSTALL.md).

## Data Preparation :hammer:
Please refer to [data preparation](dataset/README.md).

## Training and Testing :train2:
Please refer to [training guide](docs/TRAIN.md).

## Quick Demo :fire:

### [ScanNetv2](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap)

| Method | Dataset | AP | AP_50 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|:-:|
| GaPro + ISBNet | Test | 49.3 | 69.8 | 
| ''| Val | 50.6 | 69.1 | [config](configs/scannetv2/isbnet_scannetv2.yaml) | [checkpoint](https://drive.google.com/file/d/19FYLM5SzcQSKYR4r9iChauf5cleoXC_H/view?usp=drive_link) 
| GaPro + SPFormer | Test | 48.2 | 68.2 | 
| ''| Val | 51.1 | 70.4 | [config](configs/scannetv2/isbnet_scannetv2.yaml) | [checkpoint](https://drive.google.com/file/d/19FYLM5SzcQSKYR4r9iChauf5cleoXC_H/view?usp=drive_link) 


### [S3DIS](http://buildingparser.stanford.edu/dataset.html)

| Method | Dataset | AP | AP_50 | Config | Checkpoint
|:-:|:-:|:-:|:-:|:-:|:-:|
| GaPro + ISBNet | Area 5 | 56.3 | 67.5 | [config](configs/s3dis/isbnet_s3dis_area5.yaml) | [checkpoint](https://drive.google.com/file/d/1oup4nEdgsmdwnMP1TQPmoIqZ8c1RoTgA/view?usp=share_link) 

Run evaluation with pre-trained models:

```
python3 tools/test.py <path_to_config_file> <path_to_pretrain_weight>
```

## Visualization :computer:
Please refer to [visualization guide](visualization/README.md).
We provide the qualitative results of our method at [here](docs/QUALITATIVE.md)

## Acknowledgements :clap:
This repo is built upon [SpConv](https://github.com/traveller59/spconv), [ISBNet](https://github.com/VinAIResearch/ISBNet), and [SPFormer](https://github.com/sunjiahao1999/SPFormer). 

## Contacts :email:
If you have any questions or suggestions about this repo, please feel free to contact me (ductuan.ngo99@gmail.com).