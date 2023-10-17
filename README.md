# Repository for Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems

Usage examples:

**MovieLens 100k**

Augmenting using SVD

`python main.py --dataset ml-100k --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550 --SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780 --VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006

**Amazon Digital Music**

No Augmentation using MLP

`--dataset adm --model mlp --augment-training-data --SDRM-batch-size 270 --SDRM-lr 0.000063 --SDRM-epochs 45 --MLP-hidden-layers 1 --SDRM-timesteps 38 --SDRM-noise-variance-diminisher 0.7 --VAE-batch-size 310 --VAE-hidden-layer-neurons 20 --MLP-latent-neurons 20 --VAE-lr 0.0035`

<hr>

Results can also be reproduced by specifying the arguments with the `result_dict` in `main.py` on line 113

Python version 3.8 or greater was used. 

*Note: This repository contains all the code used to create the results in Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems* 

