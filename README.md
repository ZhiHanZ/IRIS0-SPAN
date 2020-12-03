# SPAN: Spatial Pyramid Attention Network forImage Manipulation Localization
This repo is the codebase for USC IRIS lab recently published ECCV paper <SPAN: Spatial Pyramid Attention Network forImage Manipulation Localization
>
This repo does not contain the pretrain step which is using the artificial image manipulate datasets created by MantraNet, It can only support to fine tune from given dataset NIST, coverage and CASIA which is mentioned in paper
## environment requirement
you need to install tensorflow
## How to evaluate on datasets
```bash
python evaluator.py
```
## How to fine tune from datasets
```bash
python train.py
```
