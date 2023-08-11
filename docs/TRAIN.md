# Training and Testing guide

## Generate pseudo-labels from 3D axis-aligned bounding box labels.

Run the script:
```
cd ./gapro
python3 gen_ps.py --save_folder dataset/scannetv2/gaussian_process_kl_pseudo_labels
```

The pseudo labels will be stored in `dataset/scannetv2/gaussian_process_kl_pseudo_labels`

## Training ISBNet with pseudo labels
1\) First navigating to ISBNet subfolder:
```
cd ISBNet/
```

2\) Pretrain the 3D Unet backbone from scratch

```
python3 tools/train.py configs/scannetv2/boxsup_isbnet_backbone_scannetv2.yaml --only_backbone --exp_name pretrain_backbone
```

3\) Train ISBNet

```
python3 tools/train.py configs/scannetv2/boxsup_isbnet_scannetv2.yaml --trainall --exp_name default
```

4\) (Optional) Self-training

Generate pointwise deep features from the current best model

```
python3 tools/export_features.py configs/scannetv2/boxsup_isbnet_scannetv2_export_feats.yaml <checkpoint_file> --save_deepfeatures_path dataset/scannetv2/pretrain_maskfeats
```

Re-generate the pseudo labels:

```
cd ./gapro
python3 gen_ps.py --save_folder dataset/scannetv2/gaussian_process_kl_deep_pseudo_labels --use_deepfeat --deepfeat_folder dataset/scannetv2/pretrain_maskfeats/
```

Change the `label_type` in config file to the new label name (`gaussian_process_kl_deep_pseudo_labels`) and re-run step #2 and #3

## Inference

1\) For evaluation (on ScanNetV2 val, S3DIS, and STPLS3D)

```
python3 tools/test.py configs/<config_file> <checkpoint_file>
```

2\) For exporting predictions (i.e., to submit results to ScanNetV2 hidden benchmark)

```
python3 tools/test.py configs/<config_file> <checkpoint_file> --out <output_dir>
```