# GMAP

This is the official implementation of **GMAP: Gaussian Multi-view Alignment via Prototypes for Unsupervised Domain Adaptation in Medical Image Segmentation**.

## Table of Contents
- [Requirements](#requirements)
- [Download](#download)
- [Train](#train)

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

You can also create the Conda environment:
```bash
conda env create -f environment.yml
conda activate gmap
```

## Download
The pre-trained models and datasets will be released later.

## Train
### 1. Dataset Preparation
Please organise the dataset according to the following structure, where each npz file stores the image and its corresponding segmentation label with the key names {image.npy, label.npy}:
```angular2
root:[data2D]
+--mmwhs
| +--ct_train
| +--ct_val
| +--ct_test
| +--mr_train
| +--mr_val
| +--mr_test
+--mmcyc
| +--CT2MR
| +--MR2CT
+--abdominal
| +--ct_train
| +--ct_val
| +--ct_test
| +--mr_train
| +--mr_val
| +--mr_test
+--abcyc
| +--CT2MR
| +--MR2CT
```

The translated-image CSV files should contain the following columns:
```angular2
s_path,s2t_paths
```

The dataset paths and label values are configured in `config.ini`.

### 2. Supervised training
Now you can start supervised source-domain training:

For CT to MR adaptation:
```angular2
python train.py --data_path ../data2D --checkpoint_path checkpoints --mode CT --stage sup --checkpoint_name sup_CT --classes 5 --batch_size 32 --sup_epochs 100 --gpu 0
```

For MR to CT adaptation:
```angular2
python train.py --data_path ../data2D --checkpoint_path checkpoints --mode MR --stage sup --checkpoint_name sup_MR --classes 5 --batch_size 32 --sup_epochs 100 --gpu 0
```

### 3. Unsupervised training
Now you can start unsupervised domain adaptation:
```angular2
python train.py --data_path ../data2D --checkpoint_path checkpoints --mode <CT or MR or ABCT or ABMR> --stage unsup --checkpoint_name <experiment name> --classes 5 --batch_size 32 --unsup_epochs 200 --contra_weight 5 --N 5 --k 0.2 --gpu 0
```

For CT to MR adaptation:
```angular2
python train.py --data_path ../data2D --checkpoint_path checkpoints --mode CT --stage unsup --checkpoint_name study_seed44_unsup_all --classes 5 --batch_size 32 --unsup_epochs 200 --contra_weight 5 --N 5 --k 0.2 --gpu 0
```

For abdominal CT to MR adaptation:
```angular2
python train.py --data_path ../data2D --checkpoint_path checkpoints --mode ABCT --stage unsup --checkpoint_name study_seed44_unsup_all --classes 5 --batch_size 32 --unsup_epochs 200 --contra_weight 5 --N 5 --k 0.2 --gpu 0
```

## Test
You can evaluate a trained model with:
```angular2
python test.py --data_path data --checkpoint_path checkpoints --train_mode CT --test_mode MR --classes 5 --gpu 0
```

## Acknowledgement
The code is based on PyTorch and MONAI.
We thank the authors for their open-sourced code.

## Social media

<p align="center"><img width="600" alt="image" src="https://github.com/BioMedIA-repo/.github/blob/052046a248d3831a599e11c85ff94cdd658c5abc/pic/wechat.png?raw=true" height=""></p> 
Welcome to follow our [Wechat official account: iBioMedInfo] and [Xiaohongshu official account: iBioMedInfo], we will share recent studies on biomedical image and bioinformation analysis there.

## Global Collaboration & Questions

**Global Collaboration:** We're on a mission to biomedical research, aiming for artificial intelligence and its
applications to biomedical image and bioinformation analysis, promoting the development of the medical community.
Collaborate with us to increase competitiveness.

**Questions:** General questions, please contact 'zlinkw@mail.nwpu.edu.cn'
