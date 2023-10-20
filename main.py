import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from train_SDRM import train_SDRM, sample_ddpm
from dataloaders import load_data, SparseDataset, sparse_batch_collate
from svd_benchmark import compute_mf_results
from mlp_benchmark import compute_mlp_results
from neural_cf_benchmark_pt import compute_neuralcf_results
import utilities

"""
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
"""

"""
Example usage:
python main.py --dataset ml-1m --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550 
--SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780 
--VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006
"""

results_dict = {
    'ml-100k': {
        'svd': """--dataset ml-100k --model svd --augment-training-data --SDRM-epochs 265 --SDRM-batch-size 550
        --SDRM-lr 0.000021 --SDRM-timesteps 83 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 2 --VAE-batch-size 780
        --VAE-hidden-layer-neurons 930 --MLP-latent-neurons 830 --VAE-lr 0.0006""",
        'mlp': """--dataset ml-100k --model mlp --augment-training-data --SDRM-epochs 200 --SDRM-batch-size 810
        --SDRM-lr 0.000052 --SDRM-timesteps 58 --SDRM-noise-variance-diminisher 1 --MLP-hidden-layers 0 --VAE-batch-size 50
        --VAE-hidden-layer-neurons 40 --MLP-latent-neurons 40 --VAE-lr 0.0034""",
        'neumf': """--dataset ml-100k --model neumf --augment-training-data --SDRM-epochs 15 --SDRM-batch-size 190
        --SDRM-lr 0.000028 --SDRM-timesteps 138 --SDRM-noise-variance-diminisher 0.2 --MLP-hidden-layers 4 --VAE-batch-size 870
        --VAE-hidden-layer-neurons 1000 --MLP-latent-neurons 950 --VAE-lr 0.001"""
    },
    'ml-1m': {
        'svd': """--dataset ml-1m --model svd --augment-training-data --SDRM-epochs 395 --SDRM-batch-size 720
        --SDRM-lr 0.000059 --SDRM-timesteps 23 --SDRM-noise-variance-diminisher 0.5 --MLP-hidden-layers 2 --VAE-batch-size 190
        --VAE-hidden-layer-neurons 600 --MLP-latent-neurons 150 --VAE-lr 0.0066""",
        'mlp': """--dataset ml-1m --model mlp --augment-training-data --SDRM-batch-size 720 --SDRM-lr 0.000059 
        --SDRM-epochs 395 --MLP-hidden-layers 1 --SDRM-timesteps 38 --SDRM-noise-variance-diminisher 0.7 
        --VAE-batch-size 310 --VAE-hidden-layer-neurons 20 --MLP-latent-neurons 20 --VAE-lr 0.0035""",
        'neumf': """--dataset ml-1m --model neumf --augment-training-data --SDRM-batch-size 830 --SDRM-lr 0.00005 
        --SDRM-epochs 140 --MLP-hidden-layers 1 --SDRM-timesteps 178 --SDRM-noise-variance-diminisher 1 
        --VAE-batch-size 540 --VAE-hidden-layer-neurons 430 --MLP-latent-neurons 300 --VAE-lr 0.004"""
    },
    'adm': {
        'svd': """--dataset adm --model svd --augment-training-data --SDRM-batch-size 930 --SDRM-lr 0.000001 
        --SDRM-epochs 60 --MLP-hidden-layers 1 --SDRM-timesteps 163 --SDRM-noise-variance-diminisher 0.3 
        --VAE-batch-size 380 --VAE-hidden-layer-neurons 210 --MLP-latent-neurons 160 --VAE-lr 0.0011""",
        'mlp': """--dataset adm --model mlp --augment-training-data --SDRM-batch-size 270 --SDRM-lr 0.000063 
        --SDRM-epochs 45 --MLP-hidden-layers 1 --SDRM-timesteps 38 --SDRM-noise-variance-diminisher 0.7 
        --VAE-batch-size 310 --VAE-hidden-layer-neurons 20 --MLP-latent-neurons 20 --VAE-lr 0.0035""",
        'neumf': """--dataset adm --model neumf --augment-training-data --SDRM-batch-size 850 --SDRM-lr 0.000013 
        --SDRM-epochs 185 --MLP-hidden-layers 5 --SDRM-timesteps 93 --SDRM-noise-variance-diminisher 1 
        --VAE-batch-size 290 --VAE-hidden-layer-neurons 40 --MLP-latent-neurons 40 --VAE-lr 0.0014"""
    },
    'alb': {
        'svd': """--dataset alb --model svd --augment-training-data --SDRM-epochs 5 --SDRM-batch-size 370 
        --SDRM-lr 0.000032 --SDRM-timesteps 68 --SDRM-noise-variance-diminisher 0.5 --MLP-hidden-layers 2 --VAE-batch-size 420
        --VAE-hidden-layer-neurons 70 --MLP-latent-neurons 20 --VAE-lr 0.0042""",
        'mlp': """--dataset alb --model mlp --augment-training-data --SDRM-batch-size 370 --SDRM-lr 0.000039 --SDRM-epochs 200
        --MLP-hidden-layers 0 --SDRM-timesteps 43 --SDRM-noise-variance-diminisher 0.2 --VAE-batch-size 340
        --VAE-hidden-layer-neurons 550 --MLP-latent-neurons 400 --VAE-lr 0.001""",
        'neumf': """--dataset alb --model neumf --augment-training-data --SDRM-batch-size 820 --SDRM-lr 0.000059 
        --SDRM-epochs 485 --MLP-hidden-layers 2 --SDRM-timesteps 33 --SDRM-noise-variance-diminisher 0.3 
        --VAE-batch-size 720 --VAE-hidden-layer-neurons 450 --MLP-latent-neurons 400 --VAE-lr 0.004"""
    }
}

if __name__ == '__main__':

    ### Load arguments from CLI
    parser = argparse.ArgumentParser(
                        prog='SDRM Reproducibility',
                        description='Run this file to reproduce results from SDRM paper')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset to run experiments on')
    parser.add_argument('--model', type=str, default='svd', help='Model to run experiments on')
    parser.add_argument('--augment-training-data', action='store_true', default=False, help='Whether to augment training data with synthetic data')

    # Arguments from SDRM hyperparamter search
    parser.add_argument('--SDRM-epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--SDRM-batch-size', type=int, default=500, help='Batch size to use for training SDRM')
    parser.add_argument('--SDRM-lr', type=float, default=0.00001, help='Learning rate to use for training SDRM')
    parser.add_argument('--SDRM-timesteps', type=int, default=50, help='Number of timesteps to use for training SDRM')
    parser.add_argument('--SDRM-noise-variance-diminisher', type=float, default=0.5, help='Noise variance diminisher to use for training SDRM')
    parser.add_argument('--MLP-hidden-layers', type=int, default=2, help='Number of hidden layers to use for training MLP')
    parser.add_argument('--VAE-batch-size', type=int, default=500, help='Batch size to use for training VAE')
    parser.add_argument('--VAE-hidden-layer-neurons', type=int, default=100, help='Number of hidden layer neurons to use for training VAE')
    parser.add_argument('--MLP-latent-neurons', type=int, default=100, help='Number of latent neurons to use for training MLP')
    parser.add_argument('--VAE-lr', type=float, default=0.00001, help='Learning rate to use for training VAE')

    # Reading args from dictionary not CLI
    arguments = None
    if False:  # Change to true if you want to load results directly from results_dict
        arguments = results_dict['alb']['svd'].split()
        # arguments.remove('--augment-training-data')  # Uncomment to do only synthetic training data

    args = parser.parse_args(args=arguments)

    # Load dataset
    TRAIN_DATA, TRAIN_PARTIAL_VALID_DATA, VALID_DATA = load_data(dataset_name=args.dataset.lower(),
                                                                 data_dir_path='./data')
    N_ITEMS = TRAIN_DATA.shape[1]
    N_USERS = TRAIN_DATA.shape[0]
    SPARSITY = 1 - (TRAIN_DATA.nnz / (TRAIN_DATA.shape[0] * TRAIN_DATA.shape[1]))

    # Create dataloaders
    ds = SparseDataset(TRAIN_PARTIAL_VALID_DATA, TRAIN_PARTIAL_VALID_DATA)
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds, generator=torch.Generator(device='cpu')),
        batch_size=args.SDRM_batch_size,
        drop_last=False)

    dl = DataLoader(ds,
                    batch_size=1,
                    collate_fn=sparse_batch_collate,
                    generator=torch.Generator(device='cpu'),
                    sampler=sampler,
                    shuffle=False)

    # Run 5 trials
    f_sdrm_results = []
    m_sdrm_results = []
    multivae_results = []
    for run_n in range(5):
        start_time = time.time()
        print(10*'#', 'Starting run', run_n+1, 10*'#')

        # Train SDRM
        SDRM, VAE = train_SDRM(
            dl=dl,
            N_ITEMS=N_ITEMS,
            VAE_LATENT=args.MLP_latent_neurons,
            VAE_HIDDEN=args.VAE_hidden_layer_neurons,
            VAE_LR=args.VAE_lr,
            VAE_BATCH_SIZE=args.VAE_batch_size,
            DIFF_LATENT=args.MLP_latent_neurons,
            DIFF_TRAINING_EPOCHS=args.SDRM_epochs,
            DIFF_LR=args.SDRM_lr,
            N_HIDDEN_MLP_LAYERS=args.MLP_hidden_layers,
            TIMESTEPS=args.SDRM_timesteps,
            noise_divider=args.SDRM_noise_variance_diminisher,
            VAE_DIR_PATH='./temp_vae',
            TRAIN_PARTIAL_VALID_DATA=TRAIN_PARTIAL_VALID_DATA,
            VALID_DATA=VALID_DATA,
            OPTIMIZATION_OBJECTIVE='Recall@10',
            verbose=True)


        # Sample Multi-resolution Data
        print('Sampling Multi-resolution Data')
        M_SDRM = sample_ddpm(N_USERS, SDRM, VAE, args.MLP_latent_neurons, args.SDRM_noise_variance_diminisher,
                                    timesteps='random', n_timesteps=args.SDRM_timesteps, verbose=True).detach().cpu().numpy()
        # Sample Full-resolution Data
        print('Sampling Full-resolution Data')
        F_SDRM = sample_ddpm(N_USERS, SDRM, VAE, args.MLP_latent_neurons, args.SDRM_noise_variance_diminisher,
                             n_timesteps=args.SDRM_timesteps, verbose=True).detach().cpu().numpy()

        threshold = np.quantile(M_SDRM.flatten(), SPARSITY)  # Get the threshold
        M_SDRM_equal_sparsity = pd.DataFrame((M_SDRM >= threshold).astype(int))
        threshold = np.quantile(F_SDRM.flatten(), SPARSITY)  # Get the threshold
        F_SDRM_equal_sparsity = pd.DataFrame((F_SDRM >= threshold).astype(int))

        # Sample from VAE
        multivae_raw = VAE.sample(N_USERS)
        threshold = np.quantile(multivae_raw.flatten(), SPARSITY)  # Get the threshold
        multivae_equal_sparsity = pd.DataFrame((multivae_raw >= threshold).astype(int))

        ### Evaluate SDRM ###
        if args.model.lower() == 'svd':
            m_recall, m_ndcg = compute_mf_results(TRAIN_DATA, VALID_DATA, synthetic_data=M_SDRM_equal_sparsity, nnmf=False,
                                              only_synthetic=args.augment_training_data)
            f_recall, f_ndcg = compute_mf_results(TRAIN_DATA, VALID_DATA, synthetic_data=F_SDRM_equal_sparsity, nnmf=False,
                                              only_synthetic=args.augment_training_data)
            vae_recall, vae_ndcg = compute_mf_results(TRAIN_DATA, VALID_DATA, synthetic_data=multivae_equal_sparsity, nnmf=False,
                                              only_synthetic=args.augment_training_data)

        ### Evaluate MLP ###
        elif args.model.lower() == 'mlp':
            if args.augment_training_data:
                temp_data = np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), M_SDRM_equal_sparsity.to_numpy()], axis=0)
            else:
                temp_data = M_SDRM_equal_sparsity.to_numpy()
            m_recall, m_ndcg = compute_mlp_results(temp_data, VALID_DATA)

            if args.augment_training_data:
                temp_data = np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), F_SDRM_equal_sparsity.to_numpy()], axis=0)
            else:
                temp_data = F_SDRM_equal_sparsity.to_numpy()
            f_recall, f_ndcg = compute_mlp_results(temp_data, VALID_DATA)

            if args.augment_training_data:
                temp_data = np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), multivae_equal_sparsity.to_numpy()], axis=0)
            else:
                temp_data = multivae_equal_sparsity.to_numpy()
            vae_recall, vae_ndcg = compute_mlp_results(temp_data, VALID_DATA)

        ### Evaluate NeuMF ###
        elif args.model.lower() == 'neumf':
            ### Format data for NeuMF ###
            row_train_data = pd.DataFrame(
                [TRAIN_DATA.tocoo().row, TRAIN_DATA.tocoo().col, TRAIN_DATA.tocoo().data]).T.sort_values(by=0)
            row_valid_data = pd.DataFrame(
                [VALID_DATA.tocoo().row, VALID_DATA.tocoo().col, VALID_DATA.tocoo().data]).T.sort_values(by=0)
            row_valid_data[0] += TRAIN_DATA.shape[
                0]  # Start the user column at the number of users in the training data

            # Splitting the validation data into two sets and ignoring the zeros (same splits as in MLP/SVD)
            valid_train, valid_test = utilities.split_train_test_proportion_from_csr_matrix(VALID_DATA, batch_size=10,
                                                                                            random_seed=123,
                                                                                            ignore_zeros=True)
            # Converting the validation dataset(s) into user-item-rating format
            row_valid_train_no_zeros = pd.DataFrame(
                [valid_train.tocoo().row, valid_train.tocoo().col, valid_train.tocoo().data]).T.sort_values(by=0)
            row_valid_train_no_zeros[0] += TRAIN_DATA.shape[0]
            # Start the user column at the number of users in the training data

            row_valid_test_no_zeros = pd.DataFrame(
                [valid_test.tocoo().row, valid_test.tocoo().col, valid_test.tocoo().data]).T.sort_values(by=0)
            row_valid_test_no_zeros[0] += TRAIN_DATA.shape[0]
            # Start the user column at the number of users in the training data

            # Get only the zero ratings from the validation dataset
            row_valid_data_only_zeros = row_valid_data[row_valid_data[2] == 0].sample(frac=1, random_state=123)
            # Combine the zero ratings with training and testing data
            row_valid_train = pd.concat(
                [row_valid_data_only_zeros[:int(row_valid_data_only_zeros.shape[0] / 2)], row_valid_train_no_zeros])
            valid_data = pd.concat(
                [row_valid_data_only_zeros[int(row_valid_data_only_zeros.shape[0] / 2):],
                 row_valid_test_no_zeros]).sample(
                frac=1, random_state=123)
            # Combine valid train with train data
            train_data = pd.concat([row_train_data, row_valid_train]).sample(frac=1, random_state=123)

            train_data.reset_index(drop=True, inplace=True)
            valid_data.reset_index(drop=True, inplace=True)
            train_data = train_data[~train_data.isin(valid_data)].dropna()
            train_data.reset_index(drop=True, inplace=True)

            ### Getting data for F-SDRM ###
            upper_threshold = np.quantile(F_SDRM.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
            lower_threshold = np.quantile(F_SDRM.flatten(), 1 - SPARSITY)
            upper_equal_sparsity = pd.DataFrame((F_SDRM >= upper_threshold).astype(int))
            lower_equal_sparsity = pd.DataFrame((F_SDRM <= lower_threshold).astype(int))
            csr_upper_equal_sparsity = csr_matrix(upper_equal_sparsity)
            csr_lower_equal_sparsity = csr_matrix(lower_equal_sparsity)
            csr_equal_sparsity_ones = pd.DataFrame(
                [csr_upper_equal_sparsity.tocoo().row, csr_upper_equal_sparsity.tocoo().col,
                 csr_upper_equal_sparsity.tocoo().data]).T.sort_values(by=0)
            csr_equal_sparsity_zeros = pd.DataFrame(
                [csr_lower_equal_sparsity.tocoo().row, csr_lower_equal_sparsity.tocoo().col,
                 csr_lower_equal_sparsity.tocoo().data]).T.sort_values(by=0)
            csr_equal_sparsity_zeros[2] = 0
            # Combine the zero ratings with training and testing data
            equal_sparsity = pd.concat([csr_equal_sparsity_zeros, csr_equal_sparsity_ones]).sample(frac=1,
                                                                                                   random_state=123)
            equal_sparsity_valid_train = pd.concat([equal_sparsity, row_valid_train]).sample(frac=1,
                                                                                             random_state=123)  # Add the validation back into the synthetic
            equal_sparsity[0] = equal_sparsity[0] + (TRAIN_DATA.shape[0] + VALID_DATA.shape[
                0])  # Start the user column at the number of users in the training data
            if args.augment_training_data:
                data = pd.concat([train_data, equal_sparsity], axis=0)
            else:
                data = equal_sparsity_valid_train
            f_recall, f_ndcg = compute_neuralcf_results(data, valid_data, n_users=int(pd.DataFrame(data)[0].max()) + 1,
                                                    n_items=int(pd.DataFrame(data)[1].max()) + 1)

            ### Getting data for M-SDRM ###
            upper_threshold = np.quantile(F_SDRM.flatten(),
                                          SPARSITY)  # Get the threshold for 0.1% sparsity
            lower_threshold = np.quantile(F_SDRM.flatten(), 1 - SPARSITY)
            upper_equal_sparsity = pd.DataFrame((F_SDRM >= upper_threshold).astype(int))
            lower_equal_sparsity = pd.DataFrame((F_SDRM <= lower_threshold).astype(int))
            csr_upper_equal_sparsity = csr_matrix(upper_equal_sparsity)
            csr_lower_equal_sparsity = csr_matrix(lower_equal_sparsity)
            # zero_threshold = pd.DataFrame((testing_data > 0).astype(int))

            # Convert the sparse matrix into user-item-rating format
            csr_equal_sparsity_ones = pd.DataFrame(
                [csr_upper_equal_sparsity.tocoo().row, csr_upper_equal_sparsity.tocoo().col,
                 csr_upper_equal_sparsity.tocoo().data]).T.sort_values(by=0)
            csr_equal_sparsity_zeros = pd.DataFrame(
                [csr_lower_equal_sparsity.tocoo().row, csr_lower_equal_sparsity.tocoo().col,
                 csr_lower_equal_sparsity.tocoo().data]).T.sort_values(by=0)
            csr_equal_sparsity_zeros[2] = 0

            # Combine the zero ratings with training and testing data
            equal_sparsity = pd.concat([csr_equal_sparsity_zeros, csr_equal_sparsity_ones]).sample(frac=1,
                                                                                                   random_state=123)
            equal_sparsity_valid_train = pd.concat([equal_sparsity, row_valid_train]).sample(frac=1,
                                                                                             random_state=123)  # Add the validation back into the synthetic
            equal_sparsity[0] = equal_sparsity[0] + (TRAIN_DATA.shape[0] + VALID_DATA.shape[0])  # Start the user column at the number of users in the training data
            if args.augment_training_data:
                data = pd.concat([train_data, equal_sparsity], axis=0)
            else:
                data = equal_sparsity_valid_train
            m_recall, m_ndcg = compute_neuralcf_results(data, valid_data, n_users=int(pd.DataFrame(data)[0].max()) + 1,
                                                    n_items=int(pd.DataFrame(data)[1].max()) + 1)

            ### Getting data for MultiVAE++ ###
            upper_threshold = np.quantile(multivae_raw.flatten(),
                                          SPARSITY)  # Get the threshold for 0.1% sparsity
            lower_threshold = np.quantile(multivae_raw.flatten(), 1 - SPARSITY)
            upper_equal_sparsity = pd.DataFrame((multivae_raw >= upper_threshold).astype(int))
            lower_equal_sparsity = pd.DataFrame((multivae_raw <= lower_threshold).astype(int))
            csr_upper_equal_sparsity = csr_matrix(upper_equal_sparsity)
            csr_lower_equal_sparsity = csr_matrix(lower_equal_sparsity)
            # zero_threshold = pd.DataFrame((testing_data > 0).astype(int))

            # Convert the sparse matrix into user-item-rating format
            csr_equal_sparsity_ones = pd.DataFrame(
                [csr_upper_equal_sparsity.tocoo().row, csr_upper_equal_sparsity.tocoo().col,
                 csr_upper_equal_sparsity.tocoo().data]).T.sort_values(by=0)
            csr_equal_sparsity_zeros = pd.DataFrame(
                [csr_lower_equal_sparsity.tocoo().row, csr_lower_equal_sparsity.tocoo().col,
                 csr_lower_equal_sparsity.tocoo().data]).T.sort_values(by=0)
            csr_equal_sparsity_zeros[2] = 0

            # Combine the zero ratings with training and testing data
            equal_sparsity = pd.concat([csr_equal_sparsity_zeros, csr_equal_sparsity_ones]).sample(frac=1,
                                                                                                   random_state=123)
            equal_sparsity_valid_train = pd.concat([equal_sparsity, row_valid_train]).sample(frac=1,
                                                                                             random_state=123)  # Add the validation back into the synthetic
            equal_sparsity[0] = equal_sparsity[0] + (TRAIN_DATA.shape[0] + VALID_DATA.shape[0])  # Start the user column at the number of users in the training data
            if args.augment_training_data:
                data = pd.concat([train_data, equal_sparsity], axis=0)
            else:
                data = equal_sparsity_valid_train
            vae_recall, vae_ndcg = compute_neuralcf_results(data, valid_data, n_users=int(pd.DataFrame(data)[0].max()) + 1,
                                                    n_items=int(pd.DataFrame(data)[1].max()) + 1)

        else:
            raise RuntimeError(f'{args.model.lower()} not a valid model. Please selection from SVD, MLP, or NeuMF')

        f_sdrm_results.append(np.concatenate([f_recall, f_ndcg]).reshape(-1, 1))
        m_sdrm_results.append(np.concatenate([m_recall, m_ndcg]).reshape(-1, 1))
        multivae_results.append(np.concatenate([vae_recall, vae_ndcg]).reshape(-1, 1))
        print('Run', run_n+1, 'took', round(time.time() - start_time, 2), 'seconds')


    # Print results
    mean_results = pd.DataFrame(np.concatenate([np.nanmean(f_sdrm_results, axis=0), np.nanmean(m_sdrm_results, axis=0),
                                                np.nanmean(multivae_results, axis=0)], axis=1),
                                index=['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                                       'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50'],
                                columns=['F-SDRM', 'M-SDRM', 'MultiVAE++']).round(4)
    max_results = pd.DataFrame(np.concatenate([np.nanmax(f_sdrm_results, axis=0), np.nanmax(m_sdrm_results, axis=0),
                                               np.nanmax(multivae_results, axis=0)], axis=1),
                                index=['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                                       'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50'],
                                columns=['F-SDRM', 'M-SDRM', 'MultiVAE++']).round(4)
    std_results = pd.DataFrame(np.concatenate([np.nanstd(f_sdrm_results, axis=0), np.nanstd(m_sdrm_results, axis=0),
                                               np.nanstd(multivae_results, axis=0)], axis=1),
                                index=['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                                       'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50'],
                                columns=['F-SDRM', 'M-SDRM', 'MultiVAE++']).round(4)
    print('\nMean\n', mean_results.to_markdown(), sep='')
    print('\nMax\n', max_results.to_markdown(), sep='')
    print('\nStandard Deviation\n', std_results.to_markdown(), sep='')










