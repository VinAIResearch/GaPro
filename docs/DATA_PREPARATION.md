# Data preparation guide

## ScanNetV2 dataset

1\) Download the [ScanNetV2](http://www.scan-net.org/) dataset.

2\) Put the downloaded ``scans`` and ``scans_test`` folder as follows.

```
GaPro
├── ISBNet
│   ├── dataset
│   │   ├── scannetv2
│   │   │   ├── scans
│   │   │   ├── scans_test
```

3\) Split and preprocess data

```
cd ./ISBNet/dataset/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
GaPro
├── ISBNet
│   ├── dataset
│   │   ├── scannetv2
│   │   │   ├── scans
│   │   │   ├── scans_test
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   │   ├── superpoints
│   │   │   ├── scans_transform
```

## S3DIS dataset

1\) Download the [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset (`v1.2_Aligned_Version`). 

2\) Download the preprocessed `superpoints` from Box2Mask: [superpoints](https://datasets.d2.mpi-inf.mpg.de/box2mask/segment_labels.tar.gz) and organize as below.

```
GaPro
├── ISBNet
│   ├── dataset
│   │   ├── s3dis
│   │   │   ├── Stanford3dDataset_v1.2_Aligned_Version
│   │   │   │   ├── Area_1
│   │   │   │   │   ├── hallway_1 
│   │   │   │   │   │   ├── Annotations # Contains instances information 
│   │   │   │   │   │   │   ├── door_2.txt 
│   │   │   │   │   │   │   ├── floor_1.txt
│   │   │   │   │   │   │   ├── wall_2.txt
│   │   │   │   │   │   │   ├── ...
│   │   │   │   │   │   ├── hallway_1.txt # Contains positions and colors of scene points
│   │   │   │   │   ├── office_1
│   │   │   │   │   ├── ...
│   │   │   │   ├── Area_2
│   │   │   │   ├── Area_3
│   │   │   │   ├── Area_4
│   │   │   │   ├── Area_5
│   │   │   │   ├── Area_6
│   │   │   ├── learned_superpoin_graph_segmentations
```


3\) Preprocess data

```
cd ./ISBNet/dataset/s3dis
bash prepare_data.sh
```

After running the script the scannet dataset structure should look like below.

```
GaPro
├── ISBNet
│   ├── dataset
│   │   ├── s3dis
│   │   │   ├── Stanford3dDataset_v1.2_Aligned_Version
│   │   │   ├── learned_superpoin_graph_segmentations
│   │   │   ├── preprocess
│   │   │   ├── superpoints
```
