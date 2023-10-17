# Repository for Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems



Usage Example:

\>\>\> python main.py --dataset ml-1m --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550 --SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780 --VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006

Results can also be reproduced by specifying the arguments with the `result_dict` in `main.py` on line 113

Python version 3.8 or greater was used. 

*Note: This repository contains all the code used to create the results in Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems* 

