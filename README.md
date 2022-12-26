# [Work in Progress] MICCAI 2022: What Makes for Automatic Reconstruction of Pulmonary Segments

This is the official implementation of the MICCAI 2022 paper: *What Makes for Automatic Reconstruction of Pulmonary Segments*, created by Kaiming Kuang\*, Li Zhang\*, Jingyu Li, Hongwei Li, Jiajun Chen, Bo Du and Jiancheng Yang\*\*.

\*: Equal contributions.  
\*\*: Corresponding author.

![Network figure](./figures/network.png)

# Citation

If you find our work useful, please consider citing as follows:
```
@inproceedings{Kuang2022WhatMF,
  title={What Makes for Automatic Reconstruction of Pulmonary Segments},
  author={Kaiming Kuang and Li Zhang and Jingyuan Li and Hongwei Li and Jiajun Chen and Bo Du and Jiancheng Yang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022}
}
```

# Usage

## ImPulSe Training
To train the proposed ImPulSe model, run:
```bash
python train.py --cfg=ibav --data_dir=/data/directory --df_path=/data/info/path --log_dir=/tensorboard/log/directory
```

## ImPulSe Inference
To run inference with the trained ImPulSe, run:
```bash
python predict.py --cfg=ibav --data_dir=/data/directory --df_path=/data/info/path --weight_path=/path/to/trained/model --output_dir=/prediction/output/directory
```

## Prediction evaluation
To evaluate your model predictions, run:
```bash
python evaluate.py --gt_dir=/ground/truth/directory --pred_dir=/prediction/directory --df_path=/data/info/path
```
