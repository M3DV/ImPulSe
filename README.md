# MICCAI 2022: What Makes for Automatic Reconstruction of Pulmonary Segments

This is the official implementation of the MICCAI 2022 paper: *What Makes for Automatic Reconstruction of Pulmonary Segments*, created by Kaiming Kuang\*, Li Zhang\*, Jingyu Li, Hongwei Li, Jiajun Chen, Bo Du and Jiancheng Yang\*\*.

\*: Equal contributions.  
\*\*: Corresponding author.

![Network figure](./figures/network.png)

# Usage

## Data Preparation
Since the ImPulSe model and the related dataset are restricted to the proprietary rights of Dianei Technologies, we may not open-source the trained model or the training dataset. However, we provide a sample dataset in `sample_data/`, which you should follow before running our scripts.

The sample training data is in `sample_data/training`. There should be multiple `.npz` files that contain the training images and labels. In each `.npz` file, there are multiple keys:
- airway: A 3D binary airway mask reshaped as `128×128×128`, cropped around the lung area.
- artery: A 3D binary artery mask reshaped as `128×128×128`, cropped around the lung area.
- vein: A 3D binary vein mask reshaped as `128×128×128`, cropped around the lung area.
- lungsegment: A 3D integer lung segment mask of the original shape of the CT image, cropped around the lung area. There should be 19 classes in total, including the background class (0) and 18 lung segments (1-18).
- image: A 3D CT image reshaped as `128×128×128`, cropped around the lung area. The original voxel values are kept.
- lobe: A 3D integer lung lobe mask reshaped as `128×128×128`, cropped around the lung area. There should be 6 classes in total, including the background class (0) and 5 lung lobes (1-5).

The sample test data is in `sample_data/inference`, which follows similar rules as the training dataset. However, there are two major differences to notice:
- All masks and images in the test data should be in the original shape of the CT image.
- An additional key `lung_bbox` is included in each `.npz` file, which indicates the lung bounding box as a matrix in shape of `3×2`. For example:
```
data["lung_bbox"]
array([[  5, 170],
       [176, 409],
       [ 87, 442]], dtype=int32)
```
means that the lung bounding box starts from `5, 176, 87` and ends at `170, 409, 442` in x, y, z axes, respectively.

Both training and test folder should include a `.csv` file as well. For example:
```
pid,subset
pulse_00001,train
pulse_00002,val
```
This `.csv` file should include two columns: `pid` and `subset`. In the training `.csv` file, the subset column should be `train` or `val`. In the test `.csv` file, the subset column should be `val` or `test`.

## ImPulSe Training
To train the proposed ImPulSe model, run:
```bash
python train.py --cfg=ibav --data_dir=sample_data/training --df_path=sample_data/training/train.csv --log_dir=logs
```
The `--data_dir` argument indicates the training data folder, and the `--df_path` argument indicates the training csv file path. The `--log_dir` argument indicates the logging directory, where the trained model weights are saved.

## ImPulSe Inference
To run inference with the trained ImPulSe, run:
```bash
python predict.py --cfg=ibav --data_dir=/data/directory --df_path=/data/info/path --weight_path=/path/to/trained/model --output_dir=/prediction/output/directory
```
The `--data_dir` argument indicates the training data folder, and the `--df_path` argument indicates the training csv file path. The `--weight_path` argument indicates the trained weight path. The `--output_dir` is where the output predictions are saved.

# Citation

If you find our work useful, please consider citing as follows:
```
@inproceedings{Kuang2022WhatMF,
  title={What Makes for Automatic Reconstruction of Pulmonary Segments},
  author={Kaiming Kuang and Li Zhang and Jingyu Li and Hongwei Li and Jiajun Chen and Bo Du and Jiancheng Yang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022}
}
```
