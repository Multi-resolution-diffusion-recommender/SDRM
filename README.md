# Repository for Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems

**Best hyperparameter results in paper**

| Dataset                   |  ML-100k |          |          |    ALB   |          |          |   ML-1M  |          |          |    ADM   |          |          |
|---------------------------|:--------:|----------|----------|:--------:|----------|----------|:--------:|----------|----------|:--------:|----------|----------|
| Recommender Model         | SVD      | MLP      | NeuMF    | SVD      | MLP      | NeuMF    | SVD      | MLP      | NeuMF    | SVD      | MLP      | NeuMF    |
| Best trial number         | 223      | 193      | 44       | 69       | 91       | 67       | 76       | 20       | 4        | 38       | 40       | 22       |
| Recall@10 score           | 0.3924   | 0.3839   | 0.232    | 0.339    | 0.3246   | 0.3225   | 0.3722   | 0.3595   | 0.1026   | 0.0651   | 0.0868   | 0.0234   |
| SDRM batch size           | 550      | 810      | 190      | 370      | 530      | 820      | 720      | 160      | 830      | 930      | 270      | 850      |
| SDRM lr                   | 2.10E-05 | 5.20E-05 | 2.80E-05 | 3.20E-05 | 3.90E-05 | 5.90E-05 | 5.90E-05 | 9.80E-05 | 5.00E-06 | 1.00E-06 | 6.30E-05 | 1.30E-05 |
| SDRM epochs               | 265      | 200      | 15       | 5        | 200      | 485      | 395      | 15       | 140      | 60       | 45       | 185      |
| MLP hidden layers         | 2        | 0        | 4        | 2        | 0        | 2        | 2        | 1        | 1        | 1        | 1        | 5        |
| SDRM timesteps            | 83       | 58       | 138      | 68       | 43       | 33       | 23       | 78       | 178      | 163      | 38       | 93       |
| VAE batch size            | 780      | 50       | 870      | 420      | 340      | 720      | 190      | 270      | 540      | 380      | 310      | 290      |
| VAE hidden layer neurons* | 930      | 40       | 1000     | 70       | 550      | 450      | 600      | 490      | 430      | 210      | 20       | 40       |
| MLP latent neurons        | 830      | 40       | 950      | 20       | 400      | 400      | 150      | 340      | 330      | 160      | 20       | 40       |
| VAE lr                    | 0.0006   | 0.0034   | 0.001    | 0.0042   | 0.001    | 0.004    | 0.0066   | 0.0002   | 0.0009   | 0.0011   | 0.0035   | 0.0014   |
| Noise variance diminisher | 1        | 1        | 0.2      | 0.5      | 0.2      | 0.3      | 0.5      | 1        | 1        | 0.3      | 0.7      | 1        |

Usage examples:

**MovieLens 100k**

Augmenting using SVD

`python main.py --dataset ml-100k --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550 --SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780 --VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006`

**Amazon Digital Music**

No Augmentation using MLP

`--dataset adm --model mlp --augment-training-data --SDRM-batch-size 270 --SDRM-lr 0.000063 --SDRM-epochs 45 --MLP-hidden-layers 1 --SDRM-timesteps 38 --SDRM-noise-variance-diminisher 0.7 --VAE-batch-size 310 --VAE-hidden-layer-neurons 20 --MLP-latent-neurons 20 --VAE-lr 0.0035`

<hr>

Results can also be reproduced by specifying the arguments with the `result_dict` in `main.py` on line 113

Python version 3.8 or greater was used. 

hyperparameter_search_results.7z in the `data` folder contains all the raw hyperparameter search results and optuna study objects.

*Note: This repository contains all the code used to create the results in Multi-Resolution Diffusion for Privacy Sensitive Recommender Systems* 

