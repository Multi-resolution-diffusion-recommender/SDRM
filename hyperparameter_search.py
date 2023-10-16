import numpy as np
from scipy.sparse import csr_matrix, vstack, coo_matrix
from torch.utils.data import DataLoader
import torch

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import math
import sys
import joblib
import shutil
import datetime

import optuna
from optuna.trial import TrialState

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

from tqdm import tqdm
import pickle
import utilities

from mlp_benchmark import compute_mlp_results
from svd_benchmark import compute_mf_results
from neural_cf_benchmark_pt import compute_neuralcf_results
from train_SDRM import train_SDRM
from dataloaders import SparseDataset, sparse_batch_collate, load_data

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
DATALOADER_DEVICE = 'cpu'

import logging
import os

BEST_SCORE = -math.inf
BEST_TRIAL = 0

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise


@torch.no_grad()
def sample_ddpm(n_sample, diff_net, vae_net, diff_latent_dim, noise_divider=1.0, timesteps=None):
    # x_T ~ N(0, 1), sample initial noise
    # samples = torch.randn(n_sample, N_ITEMS).to(DEVICE)
    diff_net.eval()
    vae_net.eval()
    with torch.no_grad():

        # Randomly sample timesteps for each sample
        if timesteps == 'random':
            encode_x = torch.randn(n_sample, diff_latent_dim).to(DEVICE)  # sample from prior, pure gaussian noise

            for j in range(n_sample):
                timesteps = np.random.randint(1, TIMESTEPS)
                for i in range(timesteps, 0, -1):

                    # sample some random noise to inject back in. For i = 1, don't add back in noise
                    z = torch.randn(diff_latent_dim).to(DEVICE) * noise_divider if i > 1 else 0
                    pred_noise_eps = diff_net.forward(torch.unsqueeze(encode_x[j], dim=0), torch.as_tensor([i], device=DEVICE)) # predict noise e_(x_t,t)
                    encode_x[j] = torch.squeeze(denoise_add_noise(torch.unsqueeze(encode_x[j], dim=0), torch.as_tensor([i], device=DEVICE), pred_noise_eps, z))
            samples = vae_net.decode(encode_x)
        else:
            encode_x = torch.randn(n_sample, diff_latent_dim).to(DEVICE)

            for i in range(timesteps, 0, -1):

                # sample some random noise to inject back in. For i = 1, don't add back in noise
                z = torch.randn_like(encode_x) * noise_divider if i > 1 else 0

                pred_noise_eps = diff_net.forward(encode_x, torch.full((n_sample,), i, device=DEVICE, dtype=torch.long))  # predict noise e_(x_t,t)
                encode_x = denoise_add_noise(encode_x, i, pred_noise_eps, z)

            samples = vae_net.decode(encode_x)
    return samples


def convert_optimization_to_num(objective):
    if 'Recall' in objective:
        col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50']
    else:
        col_names = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
    return col_names.index(objective)


def mlp_objective(trial, logger, BEST_SCORE):
    logger.info(f'Starting trial: {trial.number}')
    global beta1
    global beta2
    global b_t
    global a_t
    global ab_t

    # Optimizing hyperparameters
    global TIMESTEPS
    DIFF_TRAINING_EPOCHS = trial.suggest_int("DIFF_TRAINING_EPOCHS", 5, 501, step=5)
    DIFF_LR = trial.suggest_float("DIFF_LR", 0.000001, 0.0001, step=0.000001)
    N_HIDDEN_MLP_LAYERS = trial.suggest_int("N_HIDDEN_MLP_LAYERS", 0, 5)
    TIMESTEPS = trial.suggest_int("TIMESTEPS", 3, 200, step=5)
    noise_divider = trial.suggest_categorical("noise_divider", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01])
    VAE_LATENT = trial.suggest_int("VAE_LATENT", 20, 1000, step=10)
    VAE_HIDDEN = trial.suggest_int("VAE_HIDDEN", VAE_LATENT, 1000, step=50)
    VAE_LR = trial.suggest_float("VAE_LR", 0.0001, 0.01, step=0.0001)
    VAE_BATCH_SIZE = trial.suggest_int("VAE_BATCH_SIZE", 30, 1000, step=10)
    DIFF_LATENT = VAE_LATENT
    BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 30, 1000, step=10)

    # Create linear scheduler for beta1 and beta2
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, TIMESTEPS + 1, device=DEVICE) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    ds = SparseDataset(TRAIN_PARTIAL_VALID_DATA, TRAIN_PARTIAL_VALID_DATA)
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds, generator=torch.Generator(device=DATALOADER_DEVICE)),
        batch_size=BATCH_SIZE,
        drop_last=False)

    dl = DataLoader(ds,
                    batch_size=1,
                    collate_fn=sparse_batch_collate,
                    generator=torch.Generator(device=DATALOADER_DEVICE),
                    sampler=sampler,
                    shuffle=False)

    diff_mlp_results = []
    multivae_mlp_results = []
    metric_moving_avg = []
    if EVAL_MULTIRESOLUTION:
        diff_mlp_results_t_random = []

    # Running N amount of
    start_time = time.time()
    for run_n in tqdm(range(5)):

        # Train SDRM
        diff_mlp, variational_ae = train_SDRM(dl=dl, N_ITEMS=N_ITEMS, VAE_LATENT=VAE_LATENT, VAE_HIDDEN=VAE_HIDDEN, VAE_LR=VAE_LR,
                                              VAE_BATCH_SIZE=VAE_BATCH_SIZE, DIFF_LATENT=DIFF_LATENT,
                                              DIFF_TRAINING_EPOCHS=DIFF_TRAINING_EPOCHS, DIFF_LR=DIFF_LR,
                                              N_HIDDEN_MLP_LAYERS=N_HIDDEN_MLP_LAYERS, TIMESTEPS=TIMESTEPS,
                                              noise_divider=noise_divider, VAE_DIR_PATH=VAE_DIR_PATH,
                                              TRAIN_PARTIAL_VALID_DATA=TRAIN_PARTIAL_VALID_DATA, VALID_DATA=VALID_DATA,
                                              OPTIMIZATION_OBJECTIVE=OPTIMIZATION_OBJECTIVE)

        ###################################
        ###### Evaluation Diffusion #######
        ###################################
        synth_gen_set = sample_ddpm(N_USERS, diff_mlp, variational_ae, VAE_LATENT, noise_divider, TIMESTEPS)

        raw_predictions = synth_gen_set.cpu().numpy()

        threshold = np.quantile(raw_predictions.flatten(), SPARSITY)  # Get the threshold for sparsity of original data
        diff_datasets = {
            'Diff': raw_predictions,
            'Diff (0 Threshold)': (raw_predictions > 0).astype(int),
            'Diff (Equal Sparsity)': (raw_predictions >= threshold).astype(int),
            'Diff + Original': np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), raw_predictions], axis=0),
            'Diff (0 Threshold) + Original': np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), (raw_predictions > 0).astype(int)],
                                                            axis=0),
            'Diff (Equal Sparsity) + Original': np.concatenate(
                [TRAIN_PARTIAL_VALID_DATA.toarray(), (raw_predictions >= threshold).astype(int)], axis=0),
        }
        col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                     'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
        results = []
        for name, data in diff_datasets.items():
            recall, ndcg = compute_mlp_results(data, VALID_DATA)
            results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

        ML_MLP_RESULTS = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=diff_datasets.keys())

        # Special case for sparsity optimization
        if 'Raw' in OPTIMIZATION_SPARSITY:
            if 'Original' in OPTIMIZATION_SPARSITY:
                max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE][3]
            else:
                max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE][0]
        elif 'Equal' in OPTIMIZATION_SPARSITY:
            if 'Original' in OPTIMIZATION_SPARSITY:
                max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE][2]
            else:
                max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE][5]
        elif 'Zero' in OPTIMIZATION_SPARSITY:
            if 'Original' in OPTIMIZATION_SPARSITY:
                max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE][1]
            else:
                max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE][4]
        else:
            max_recall = ML_MLP_RESULTS.loc[OPTIMIZATION_OBJECTIVE].max()
        metric_moving_avg.append(max_recall)
        trial.report(np.mean(metric_moving_avg), run_n)

        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned with {np.mean(metric_moving_avg)} after {run_n} step(s): {trial.params}")
            raise optuna.TrialPruned()

        diff_mlp_results.append(ML_MLP_RESULTS)
        ML_MLP_RESULTS.to_csv(INTERMEDIATE_TRIAL_RESULTS + f'_diff_trial_{trial.number}_run_{run_n}.csv', index=True)

        #### Evaluate Multiresolution Sampling ####
        if EVAL_MULTIRESOLUTION:
            synth_gen_set_t_random = sample_ddpm(N_USERS, diff_mlp, variational_ae, VAE_LATENT, noise_divider, 'random').cpu().numpy()

            for idx, sampled_data in enumerate([synth_gen_set_t_random]):
                threshold = np.quantile(sampled_data.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
                diff_datasets = {
                    'Diff': sampled_data,
                    'Diff (0 Threshold)': (sampled_data > 0).astype(int),
                    'Diff (Equal Sparsity)': (sampled_data >= threshold).astype(int),
                    'Diff + Original': np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), sampled_data], axis=0),
                    'Diff (0 Threshold) + Original': np.concatenate(
                        [TRAIN_PARTIAL_VALID_DATA.toarray(), (sampled_data > 0).astype(int)],
                        axis=0),
                    'Diff (Equal Sparsity) + Original': np.concatenate(
                        [TRAIN_PARTIAL_VALID_DATA.toarray(), (sampled_data >= threshold).astype(int)], axis=0),
                }
                col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                             'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
                results = []
                for name, data in diff_datasets.items():
                    recall, ndcg = compute_mlp_results(data, VALID_DATA)
                    results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

                multires_mlp_results = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=diff_datasets.keys())
                multires_mlp_results.to_csv(INTERMEDIATE_TRIAL_RESULTS + f'_multiresolution_diff_trial_{trial.number}_run_{run_n}.csv', index=True)
                diff_mlp_results_t_random.append(multires_mlp_results)

        ##################################
        ###### Evaluation MultiVAE #######
        ##################################
        if EVAL_MULTIVAE:
            raw_predictions = variational_ae.sample(N_USERS)
            threshold = np.quantile(raw_predictions.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
            multivae_datasets = {
                #'Original': TRAIN_PARTIAL_VALID_DATA,
                'MultiVAE': raw_predictions,
                'MultiVAE (0 Threshold)': (raw_predictions > 0).astype(int),
                'MultiVAE (Equal Sparsity)': (raw_predictions >= threshold).astype(int),
                'MultiVAE + Original': np.concatenate([TRAIN_PARTIAL_VALID_DATA.toarray(), raw_predictions], axis=0),
                'MultiVAE (0 Threshold) + Original': np.concatenate(
                    [TRAIN_PARTIAL_VALID_DATA.toarray(), (raw_predictions > 0).astype(int)],
                    axis=0),
                'MultiVAE (Equal Sparsity) + Original': np.concatenate(
                    [TRAIN_PARTIAL_VALID_DATA.toarray(), (raw_predictions >= threshold).astype(int)], axis=0),
            }
            col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                         'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
            results = []
            for name, data in multivae_datasets.items():
                recall, ndcg = compute_mlp_results(data, VALID_DATA)
                results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

            ML_MLP_RESULTS = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=multivae_datasets.keys())
            ML_MLP_RESULTS.to_csv(INTERMEDIATE_TRIAL_RESULTS + f'_multivae_trial_{trial.number}_run_{run_n}.csv', index=True)
            multivae_mlp_results.append(ML_MLP_RESULTS)

    # Print Diffusion results
    diff_average_mlp_results = pd.DataFrame(np.round(np.mean([res.values for res in diff_mlp_results], axis=(0)), 4),
                                       index=col_names, columns=diff_datasets.keys())
    diff_max_mlp_results = pd.DataFrame(np.round(np.max([res.values for res in diff_mlp_results], axis=(0)), 4), index=col_names,
                                   columns=diff_datasets.keys())
    diff_std_mlp_results = pd.DataFrame(np.round(np.std([res.values for res in diff_mlp_results], axis=(0)), 4), index=col_names,
                                   columns=diff_datasets.keys())
    diff_average_mlp_results.to_csv(CSV_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
    diff_max_mlp_results.to_csv(CSV_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
    diff_std_mlp_results.to_csv(CSV_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)
    current_score = diff_average_mlp_results.loc[OPTIMIZATION_OBJECTIVE][5]  # 0 = Diff, 1 = Diff (0 Threshold), 2 = Diff (Equal Sparsity),3 = Diff + Original, 4 = Diff (0 Threshold) + Original, 5 = Diff (Equal Sparsity) + Original
    if current_score > BEST_SCORE:
        BEST_SCORE = current_score
        BEST_TRIAL = trial.number
    logger.info('Finished trial: ' + str(trial.number) + f'with score {current_score} and parameters {trial.params}. Best trial is {BEST_TRIAL} with score {BEST_SCORE}')
    logger.info(f'Time taken: {time.time() - start_time} seconds')

    logger.info(f'Mean\n{diff_average_mlp_results.to_markdown()}')
    logger.info(f'Max\n{diff_max_mlp_results.to_markdown()}')
    logger.info(f'Standard Dev\n{diff_std_mlp_results.to_markdown()}')

    ### Print MultiVAE results
    if EVAL_MULTIVAE:
        multivae_average_mlp_results = pd.DataFrame(np.round(np.mean([res.values for res in multivae_mlp_results], axis=(0)), 4),
                                           index=col_names, columns=multivae_datasets.keys())
        multivae_max_mlp_results = pd.DataFrame(np.round(np.max([res.values for res in multivae_mlp_results], axis=(0)), 4), index=col_names,
                                       columns=multivae_datasets.keys())
        multivae_std_mlp_results = pd.DataFrame(np.round(np.std([res.values for res in multivae_mlp_results], axis=(0)), 4), index=col_names,
                                       columns=multivae_datasets.keys())
        logger.info(f'Mean\n{multivae_average_mlp_results.to_markdown()}')
        logger.info(f'Max\n{multivae_max_mlp_results.to_markdown()}')
        logger.info(f'Standard Dev\n{multivae_std_mlp_results.to_markdown()}')
        multivae_average_mlp_results.to_csv(MULTIVAE_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
        multivae_max_mlp_results.to_csv(MULTIVAE_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
        multivae_std_mlp_results.to_csv(MULTIVAE_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)

    # evaluate diffusion at different timesteps
    if EVAL_MULTIRESOLUTION:
        diff_mlp_results_t_random_avg = pd.DataFrame(np.round(np.mean([res.values for res in diff_mlp_results_t_random], axis=(0)), 4), index=col_names, columns=diff_datasets.keys())
        diff_mlp_results_t_random_max = pd.DataFrame(np.round(np.max([res.values for res in diff_mlp_results_t_random], axis=(0)), 4), index=col_names, columns=diff_datasets.keys())
        diff_mlp_results_t_random_std = pd.DataFrame(np.round(np.std([res.values for res in diff_mlp_results_t_random], axis=(0)), 4), index=col_names, columns=diff_datasets.keys())
        logger.info(f'Random Timestep Mean: \n{diff_mlp_results_t_random_avg.to_markdown()}')
        logger.info(f'Random Timestep Max: \n{diff_mlp_results_t_random_max.to_markdown()}')
        logger.info(f'Random Timestep STD: \n{diff_mlp_results_t_random_std.to_markdown()}')
        diff_mlp_results_t_random_avg.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_t_random_mean_trial_{trial.number}.csv', index=True)
        diff_mlp_results_t_random_max.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_t_random_max_trial_{trial.number}.csv', index=True)
        diff_mlp_results_t_random_std.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_t_random_std_trial_{trial.number}.csv', index=True)
    return current_score  # Returns best average recall@50


def mf_objective(trial, logger, BEST_SCORE):
    logger.info(f'Starting trial: {trial.number}')
    global beta1
    global beta2
    global b_t
    global a_t
    global ab_t

    # Optimizing hyperparameters
    global TIMESTEPS
    DIFF_TRAINING_EPOCHS = trial.suggest_int("DIFF_TRAINING_EPOCHS", 5, 501, step=5)
    DIFF_LR = trial.suggest_float("DIFF_LR", 0.000001, 0.0001, step=0.000001)
    N_HIDDEN_MLP_LAYERS = trial.suggest_int("N_HIDDEN_MLP_LAYERS", 0, 5)
    TIMESTEPS = trial.suggest_int("TIMESTEPS", 3, 200, step=5)
    noise_divider = trial.suggest_categorical("noise_divider", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01])
    VAE_LATENT = trial.suggest_int("VAE_LATENT", 20, 1000, step=10)
    VAE_HIDDEN = trial.suggest_int("VAE_HIDDEN", VAE_LATENT, 1000, step=50)
    VAE_LR = trial.suggest_float("VAE_LR", 0.0001, 0.01, step=0.0001)
    VAE_BATCH_SIZE = trial.suggest_int("VAE_BATCH_SIZE", 30, 1000, step=10)
    DIFF_LATENT = VAE_LATENT
    BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 30, 1000, step=10)

    # Create linear scheduler for beta1 and beta2
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, TIMESTEPS + 1, device=DEVICE) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    ds = SparseDataset(TRAIN_PARTIAL_VALID_DATA, TRAIN_PARTIAL_VALID_DATA)
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds, generator=torch.Generator(device=DATALOADER_DEVICE)),
        batch_size=BATCH_SIZE,
        drop_last=False)

    dl = DataLoader(ds,
                    batch_size=1,
                    collate_fn=sparse_batch_collate,
                    generator=torch.Generator(device=DATALOADER_DEVICE),
                    sampler=sampler,
                    shuffle=False)

    diff_mf_zero_threshold_recall = []
    diff_mf_zero_threshold_ndcg = []
    diff_mf_equal_sparsity_recall = []
    diff_mf_equal_sparsity_ndcg = []
    diff_mf_raw_logits_recall = []
    diff_mf_raw_logits_ndcg = []
    multivae_mf_zero_threshold_recall = []
    multivae_mf_zero_threshold_ndcg = []
    multivae_mf_equal_sparsity_recall = []
    multivae_mf_equal_sparsity_ndcg = []
    multivae_mf_raw_logits_recall = []
    multivae_mf_raw_logits_ndcg = []
    diff_svd_multiresolution_zero_threshold_t_random_recall = []
    diff_svd_multiresolution_equal_sparsity_t_random_recall = []
    diff_svd_multiresolution_raw_logits_t_random_recall = []
    diff_svd_multiresolution_zero_threshold_t_random_ndcg = []
    diff_svd_multiresolution_equal_sparsity_t_random_ndcg = []
    diff_svd_multiresolution_raw_logits_t_random_ndcg = []
    metric_moving_avg = []

    start_time = time.time()
    # Running N amount of
    for run_n in tqdm(range(5)):

        # Training SDRM
        diff_mlp, variational_ae = train_SDRM(dl=dl, N_ITEMS=N_ITEMS, VAE_LATENT=VAE_LATENT, VAE_HIDDEN=VAE_HIDDEN, VAE_LR=VAE_LR,
                                              VAE_BATCH_SIZE=VAE_BATCH_SIZE, DIFF_LATENT=DIFF_LATENT,
                                              DIFF_TRAINING_EPOCHS=DIFF_TRAINING_EPOCHS, DIFF_LR=DIFF_LR,
                                              N_HIDDEN_MLP_LAYERS=N_HIDDEN_MLP_LAYERS, TIMESTEPS=TIMESTEPS,
                                              noise_divider=noise_divider, VAE_DIR_PATH=VAE_DIR_PATH,
                                              TRAIN_PARTIAL_VALID_DATA=TRAIN_PARTIAL_VALID_DATA, VALID_DATA=VALID_DATA,
                                              OPTIMIZATION_OBJECTIVE=OPTIMIZATION_OBJECTIVE)

        ###################################
        ###### Evaluation Diffusion #######
        ###################################

        # Sample DDPM
        synth_gen_set = sample_ddpm(N_USERS, diff_mlp, variational_ae, VAE_LATENT, noise_divider, TIMESTEPS)
        raw_predictions = synth_gen_set.cpu().numpy()
        recall_n = convert_optimization_to_num(OPTIMIZATION_OBJECTIVE)

        # Raw Logits results
        raw_recall_results, raw_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, raw_predictions,
                                                          nnmf=True if ALGORITHM_NAME == 'NMF' else False)
        diff_mf_raw_logits_recall.append(raw_recall_results)
        diff_mf_raw_logits_ndcg.append(raw_ndcg_results)
        if 'Recall' in OPTIMIZATION_OBJECTIVE:
            raw_results = raw_recall_results[recall_n]
        else:
            raw_results = raw_ndcg_results[recall_n]

        # Equal Sparsity results
        threshold = np.quantile(raw_predictions.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
        predictions = (raw_predictions >= threshold).astype(int)
        equal_recall_results, equal_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, predictions,
                                                          nnmf=True if ALGORITHM_NAME == 'NMF' else False)
        diff_mf_equal_sparsity_recall.append(equal_recall_results)
        diff_mf_equal_sparsity_ndcg.append(equal_ndcg_results)
        if 'Recall' in OPTIMIZATION_OBJECTIVE:
            equal_results = equal_recall_results[recall_n]
        else:
            equal_results = equal_ndcg_results[recall_n]

        # 0 Threshold results
        predictions = (raw_predictions > 0).astype(int)
        zero_recall_results, zero_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, predictions,
                                                          nnmf=True if ALGORITHM_NAME == 'NMF' else False)
        diff_mf_zero_threshold_recall.append(zero_recall_results)
        diff_mf_zero_threshold_ndcg.append(zero_ndcg_results)
        if 'Recall' in OPTIMIZATION_OBJECTIVE:
            zero_results = zero_recall_results[recall_n]
        else:
            zero_results = zero_ndcg_results[recall_n]

        row_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                     'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']

        # Check for prunning
        if 'Raw' in OPTIMIZATION_SPARSITY:
            max_recall = raw_results
        elif 'Equal' in OPTIMIZATION_SPARSITY:
            max_recall = equal_results
        elif 'Zero' in OPTIMIZATION_SPARSITY:
            max_recall = zero_results
        else:
            max_recall = max(raw_results, equal_results, zero_results)
        metric_moving_avg.append(max_recall)
        trial.report(np.mean(metric_moving_avg), run_n)
        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned with {np.mean(metric_moving_avg)} after {run_n} step(s): {trial.params}")
            raise optuna.TrialPruned()
        column_names = ['Diff Raw Logits', 'Diff Zero Threshold', 'Diff Equal Sparsity']
        diff_np = np.stack([np.concatenate([raw_recall_results, raw_ndcg_results]),
                                np.concatenate([zero_recall_results, zero_ndcg_results]),
                                np.concatenate([equal_recall_results, equal_ndcg_results])]).T
        pd.DataFrame(diff_np, columns=column_names, index=row_names).to_csv(
            INTERMEDIATE_TRIAL_RESULTS + f'_diff_trial_{trial.number}_run_{run_n}.csv', index=True)

        #### Evaluate Multiresolution Sampling ####
        if EVAL_MULTIRESOLUTION:
            column_names = ['Diff Multiresolution Raw Logits Random', 'Diff Multiresolution Zero Threshold Random',
                            'Diff Multiresolution Equal Sparsity Random']
            synth_gen_set_t_random = sample_ddpm(N_USERS, diff_mlp, variational_ae, VAE_LATENT, noise_divider,
                                                 'random').cpu().numpy()

            for idx, sampled_data in enumerate([synth_gen_set_t_random]):
                raw_predictions = sampled_data

                # Raw Logits results
                raw_recall_results, raw_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, raw_predictions,
                                                                          nnmf=True if ALGORITHM_NAME == 'NMF' else False)

                # Equal Sparsity results
                threshold = np.quantile(raw_predictions.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
                predictions = (raw_predictions >= threshold).astype(int)
                equal_recall_results, equal_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, predictions,
                                                                              nnmf=True if ALGORITHM_NAME == 'NMF' else False)

                # 0 Threshold results
                predictions = (raw_predictions > 0).astype(int)
                zero_recall_results, zero_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, predictions,
                                                                            nnmf=True if ALGORITHM_NAME == 'NMF' else False)

                diff_svd_multiresolution_raw_logits_t_random_recall.append(raw_recall_results)
                diff_svd_multiresolution_raw_logits_t_random_ndcg.append(raw_ndcg_results)
                diff_svd_multiresolution_zero_threshold_t_random_recall.append(zero_recall_results)
                diff_svd_multiresolution_zero_threshold_t_random_ndcg.append(zero_ndcg_results)
                diff_svd_multiresolution_equal_sparsity_t_random_recall.append(equal_recall_results)
                diff_svd_multiresolution_equal_sparsity_t_random_ndcg.append(equal_ndcg_results)
                multires_np = np.stack([np.concatenate([raw_recall_results, raw_ndcg_results]), np.concatenate([zero_recall_results, zero_ndcg_results]), np.concatenate([equal_recall_results, equal_ndcg_results])]).T
                pd.DataFrame(multires_np, columns=column_names, index=row_names).to_csv(
                    INTERMEDIATE_TRIAL_RESULTS + f'_multiresolution_diff_trial_{trial.number}_run_{run_n}.csv', index=True)


        ##################################
        ###### Evaluation MultiVAE #######
        ##################################
        if EVAL_MULTIVAE:
            column_names = ['MultiVAE Raw Logits', 'MultiVAE Zero Threshold', 'MultiVAE Equal Sparsity']
            raw_predictions = variational_ae.sample(N_USERS)

            # Raw Logits results
            raw_recall_results, raw_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, raw_predictions,
                                                              nnmf=True if ALGORITHM_NAME == 'NMF' else False)
            multivae_mf_raw_logits_recall.append(raw_recall_results)
            multivae_mf_raw_logits_ndcg.append(raw_ndcg_results)

            # Equal Sparsity results
            threshold = np.quantile(raw_predictions.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
            predictions = (raw_predictions >= threshold).astype(int)
            equal_recall_results, equal_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, predictions,
                                                              nnmf=True if ALGORITHM_NAME == 'NMF' else False)
            multivae_mf_equal_sparsity_recall.append(equal_recall_results)
            multivae_mf_equal_sparsity_ndcg.append(equal_ndcg_results)

            # 0 Threshold results
            predictions = (raw_predictions > 0).astype(int)
            zero_recall_results, zero_ndcg_results = compute_mf_results(TRAIN_DATA, VALID_DATA, predictions,
                                                              nnmf=True if ALGORITHM_NAME == 'NMF' else False)
            multivae_mf_zero_threshold_recall.append(zero_recall_results)
            multivae_mf_zero_threshold_ndcg.append(zero_ndcg_results)

            multires_np = np.stack([np.concatenate([raw_recall_results, raw_ndcg_results]),
                                    np.concatenate([zero_recall_results, zero_ndcg_results]),
                                    np.concatenate([equal_recall_results, equal_ndcg_results])]).T
            pd.DataFrame(multires_np, columns=column_names, index=row_names).to_csv(
                INTERMEDIATE_TRIAL_RESULTS + f'_multivae_trial_{trial.number}_run_{run_n}.csv', index=True)

    column_names = ['Diff Raw Logits', 'Diff Zero Threshold', 'Diff Equal Sparsity']
    diff_avg_mf_results = pd.DataFrame(np.round(
        [np.concatenate([np.mean(diff_mf_raw_logits_recall, axis=0), np.mean(diff_mf_raw_logits_ndcg, axis=0)]),
         np.concatenate([np.mean(diff_mf_zero_threshold_recall, axis=0), np.mean(diff_mf_zero_threshold_ndcg, axis=0)]),
         np.concatenate([np.mean(diff_mf_equal_sparsity_recall, axis=0), np.mean(diff_mf_equal_sparsity_ndcg, axis=0)])],
        4),
        columns=row_names, index=column_names).T
    diff_max_mf_results = pd.DataFrame(
        np.round([np.concatenate([np.max(diff_mf_raw_logits_recall, axis=0), np.max(diff_mf_raw_logits_ndcg, axis=0)]),
                  np.concatenate(
                      [np.max(diff_mf_zero_threshold_recall, axis=0), np.max(diff_mf_zero_threshold_ndcg, axis=0)]),
                  np.concatenate(
                      [np.max(diff_mf_equal_sparsity_recall, axis=0), np.max(diff_mf_equal_sparsity_ndcg, axis=0)])],
                 4),
        columns=row_names, index=column_names).T
    diff_std_mf_results = pd.DataFrame(
        np.round([np.concatenate([np.std(diff_mf_raw_logits_recall, axis=0), np.std(diff_mf_raw_logits_ndcg, axis=0)]),
                  np.concatenate(
                      [np.std(diff_mf_zero_threshold_recall, axis=0), np.std(diff_mf_zero_threshold_ndcg, axis=0)]),
                  np.concatenate(
                      [np.std(diff_mf_equal_sparsity_recall, axis=0), np.std(diff_mf_equal_sparsity_ndcg, axis=0)])],
                 4),
        columns=row_names, index=column_names).T
    diff_avg_mf_results.to_csv(CSV_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
    diff_max_mf_results.to_csv(CSV_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
    diff_std_mf_results.to_csv(CSV_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)
    current_score = diff_avg_mf_results.loc[OPTIMIZATION_OBJECTIVE][2]  # TODO add option for sparsity

    if current_score > BEST_SCORE:
        BEST_SCORE = current_score
        BEST_TRIAL = trial.number
    logger.info('Finished trial: ' + str(trial.number) + f' with score {current_score} and parameters {trial.params}. Best trial is {BEST_TRIAL} with score {BEST_SCORE}')
    logger.info(f'Time taken: {time.time() - start_time} seconds')
    logger.info(f'Mean\n{diff_avg_mf_results.to_markdown()}')
    logger.info(f'Max\n{diff_max_mf_results.to_markdown()}')
    logger.info(f'Standard Dev\n{diff_std_mf_results.to_markdown()}')

    ### Print out the MultiVAE results
    if EVAL_MULTIVAE:
        column_names = ['MultiVAE Raw Logits', 'MultiVAE Zero Threshold', 'MultiVAE Equal Sparsity']
        multivae_avg_mf_results = pd.DataFrame(np.round(
            [np.concatenate([np.mean(multivae_mf_raw_logits_recall, axis=0), np.mean(multivae_mf_raw_logits_ndcg, axis=0)]),
             np.concatenate([np.mean(multivae_mf_zero_threshold_recall, axis=0), np.mean(multivae_mf_zero_threshold_ndcg, axis=0)]),
             np.concatenate([np.mean(multivae_mf_equal_sparsity_recall, axis=0), np.mean(multivae_mf_equal_sparsity_ndcg, axis=0)])],
            4),
            columns=row_names, index=column_names).T
        multivae_max_mf_results = pd.DataFrame(
            np.round([np.concatenate([np.max(multivae_mf_raw_logits_recall, axis=0), np.max(multivae_mf_raw_logits_ndcg, axis=0)]),
                      np.concatenate(
                          [np.max(multivae_mf_zero_threshold_recall, axis=0), np.max(multivae_mf_zero_threshold_ndcg, axis=0)]),
                      np.concatenate(
                          [np.max(multivae_mf_equal_sparsity_recall, axis=0), np.max(multivae_mf_equal_sparsity_ndcg, axis=0)])],
                     4),
            columns=row_names, index=column_names).T
        multivae_std_mf_results = pd.DataFrame(
            np.round([np.concatenate([np.std(multivae_mf_raw_logits_recall, axis=0), np.std(multivae_mf_raw_logits_ndcg, axis=0)]),
                      np.concatenate(
                          [np.std(multivae_mf_zero_threshold_recall, axis=0), np.std(multivae_mf_zero_threshold_ndcg, axis=0)]),
                      np.concatenate(
                          [np.std(multivae_mf_equal_sparsity_recall, axis=0), np.std(multivae_mf_equal_sparsity_ndcg, axis=0)])],
                     4),
            columns=row_names, index=column_names).T

        logger.info(f'Mean\n{multivae_avg_mf_results.to_markdown()}')
        logger.info(f'Max\n{multivae_max_mf_results.to_markdown()}')
        logger.info(f'Standard Dev\n{multivae_std_mf_results.to_markdown()}')
        multivae_avg_mf_results.to_csv(MULTIVAE_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
        multivae_max_mf_results.to_csv(MULTIVAE_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
        multivae_std_mf_results.to_csv(MULTIVAE_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)

    ### Print out the multiresolution sampling
    if EVAL_MULTIRESOLUTION:

        column_names = ['Diff Multiresolution Raw Logits Random', 'Diff Multiresolution Zero Threshold Random', 'Diff Multiresolution Equal Sparsity Random']
        diff_t_random_avg_mf_results = pd.DataFrame(np.round(
            [np.concatenate([np.mean(diff_svd_multiresolution_raw_logits_t_random_recall, axis=0), np.mean(diff_svd_multiresolution_raw_logits_t_random_ndcg, axis=0)]),
             np.concatenate([np.mean(diff_svd_multiresolution_zero_threshold_t_random_recall, axis=0), np.mean(diff_svd_multiresolution_zero_threshold_t_random_ndcg, axis=0)]),
             np.concatenate([np.mean(diff_svd_multiresolution_equal_sparsity_t_random_recall, axis=0), np.mean(diff_svd_multiresolution_equal_sparsity_t_random_ndcg, axis=0)])],
            4),
            columns=row_names, index=column_names).T
        diff_t_random_max_mf_results = pd.DataFrame(
            np.round([np.concatenate([np.max(diff_svd_multiresolution_raw_logits_t_random_recall, axis=0), np.max(diff_svd_multiresolution_raw_logits_t_random_ndcg, axis=0)]),
                      np.concatenate(
                          [np.max(diff_svd_multiresolution_zero_threshold_t_random_recall, axis=0), np.max(diff_svd_multiresolution_zero_threshold_t_random_ndcg, axis=0)]),
                      np.concatenate(
                          [np.max(diff_svd_multiresolution_equal_sparsity_t_random_recall, axis=0), np.max(diff_svd_multiresolution_equal_sparsity_t_random_ndcg, axis=0)])],
                     4),
            columns=row_names, index=column_names).T
        diff_t_random_std_mf_results = pd.DataFrame(
            np.round([np.concatenate([np.std(diff_svd_multiresolution_raw_logits_t_random_recall, axis=0), np.std(diff_svd_multiresolution_raw_logits_t_random_ndcg, axis=0)]),
                      np.concatenate(
                          [np.std(diff_svd_multiresolution_zero_threshold_t_random_recall, axis=0), np.std(diff_svd_multiresolution_zero_threshold_t_random_ndcg, axis=0)]),
                      np.concatenate(
                          [np.std(diff_svd_multiresolution_equal_sparsity_t_random_recall, axis=0), np.std(diff_svd_multiresolution_equal_sparsity_t_random_ndcg, axis=0)])],
                     4),
            columns=row_names, index=column_names).T

        logger.info(f'Mean T Random\n{diff_t_random_avg_mf_results.to_markdown()}')
        logger.info(f'Max T Random\n{diff_t_random_max_mf_results.to_markdown()}')
        logger.info(f'Standard Dev T Random\n{diff_t_random_std_mf_results.to_markdown()}')
        diff_t_random_avg_mf_results.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
        diff_t_random_max_mf_results.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
        diff_t_random_std_mf_results.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)

    return current_score  # Returns best average recall@50


def neucf_objective(trial, logger, BEST_SCORE):
    logger.info(f'Starting trial: {trial.number}')
    global beta1
    global beta2
    global b_t
    global a_t
    global ab_t

    # Optimizing hyperparameters
    global TIMESTEPS
    DIFF_TRAINING_EPOCHS = trial.suggest_int("DIFF_TRAINING_EPOCHS", 5, 501, step=5)
    DIFF_LR = trial.suggest_float("DIFF_LR", 0.000001, 0.0001, step=0.000001)
    N_HIDDEN_MLP_LAYERS = trial.suggest_int("N_HIDDEN_MLP_LAYERS", 0, 5)
    TIMESTEPS = trial.suggest_int("TIMESTEPS", 3, 200, step=5)
    noise_divider = trial.suggest_float("noise_divider", 0.1, 1.1, step=0.1)
    VAE_LATENT = trial.suggest_int("VAE_LATENT", 20, 1000, step=10)
    VAE_HIDDEN = trial.suggest_int("VAE_HIDDEN", VAE_LATENT, 1000, step=50)
    VAE_LR = trial.suggest_float("VAE_LR", 0.0001, 0.01, step=0.0001)
    VAE_BATCH_SIZE = trial.suggest_int("VAE_BATCH_SIZE", 30, 1000, step=10)
    DIFF_LATENT = VAE_LATENT
    BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 30, 1000, step=10)

    # Create linear scheduler for beta1 and beta2
    beta1 = 1e-4
    beta2 = 0.02
    b_t = (beta2 - beta1) * torch.linspace(0, 1, TIMESTEPS + 1, device=DEVICE) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    ds = SparseDataset(TRAIN_PARTIAL_VALID_DATA, TRAIN_PARTIAL_VALID_DATA)
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(ds, generator=torch.Generator(device=DATALOADER_DEVICE)),
        batch_size=BATCH_SIZE,
        drop_last=False)

    dl = DataLoader(ds,
                    batch_size=1,
                    collate_fn=sparse_batch_collate,
                    generator=torch.Generator(device=DATALOADER_DEVICE),
                    sampler=sampler,
                    shuffle=False)

    diff_neucf_results = []
    multivae_neucf_results = []
    metric_moving_avg = []
    if EVAL_MULTIRESOLUTION:
        diff_neucf_results_t_random = []

    # Running N amount of
    start_time = time.time()
    for run_n in tqdm(range(5)):

        diff_neucf, variational_ae = train_SDRM(dl=dl, N_ITEMS=N_ITEMS, VAE_LATENT=VAE_LATENT, VAE_HIDDEN=VAE_HIDDEN, VAE_LR=VAE_LR,
                                              VAE_BATCH_SIZE=VAE_BATCH_SIZE, DIFF_LATENT=DIFF_LATENT,
                                              DIFF_TRAINING_EPOCHS=DIFF_TRAINING_EPOCHS, DIFF_LR=DIFF_LR,
                                              N_HIDDEN_MLP_LAYERS=N_HIDDEN_MLP_LAYERS, TIMESTEPS=TIMESTEPS,
                                              noise_divider=noise_divider, VAE_DIR_PATH=VAE_DIR_PATH,
                                              TRAIN_PARTIAL_VALID_DATA=TRAIN_PARTIAL_VALID_DATA, VALID_DATA=VALID_DATA,
                                              OPTIMIZATION_OBJECTIVE=OPTIMIZATION_OBJECTIVE)

        ###################################
        ###### Evaluation Diffusion #######
        ###################################
        synth_gen_set = sample_ddpm(N_USERS, diff_neucf, variational_ae, VAE_LATENT, noise_divider, TIMESTEPS)

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

        raw_predictions = synth_gen_set.cpu().numpy()
        upper_threshold = np.quantile(raw_predictions.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
        lower_threshold = np.quantile(raw_predictions.flatten(), 1 - SPARSITY)
        upper_equal_sparsity = pd.DataFrame((raw_predictions >= upper_threshold).astype(int))
        lower_equal_sparsity = pd.DataFrame((raw_predictions <= lower_threshold).astype(int))
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
        equal_sparsity = pd.concat([csr_equal_sparsity_zeros, csr_equal_sparsity_ones]).sample(frac=1, random_state=123)
        equal_sparsity_valid_train = pd.concat([equal_sparsity, row_valid_train]).sample(frac=1,
                                                                                         random_state=123)  # Add the validation back into the synthetic
        equal_sparsity[0] = equal_sparsity[0] + (TRAIN_DATA.shape[0] + VALID_DATA.shape[
            0])  # Start the user column at the number of users in the training data

        diff_datasets = {
            # 'Original': TRAIN_PARTIAL_VALID_DATA,
            'Diff': equal_sparsity_valid_train,
            'Diff + Original': pd.concat([train_data, equal_sparsity], axis=0)
        }
        col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                     'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
        results = []
        for name, data in diff_datasets.items():
            recall, ndcg = compute_neuralcf_results(data, valid_data, n_users=int(pd.DataFrame(data)[0].max()) + 1,
                                                    n_items=int(pd.DataFrame(data)[1].max()) + 1)
            results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

        ML_NEUCF_RESULTS = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=diff_datasets.keys())

        # Special case for sparsity optimization
        max_recall = ML_NEUCF_RESULTS.loc[OPTIMIZATION_OBJECTIVE][1]
        metric_moving_avg.append(max_recall)

        trial.report(np.mean(metric_moving_avg), run_n)

        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned with {np.mean(metric_moving_avg)} after {run_n} step(s): {trial.params}")
            raise optuna.TrialPruned()

        ML_NEUCF_RESULTS.to_csv(INTERMEDIATE_TRIAL_RESULTS + f'_diff_trial_{trial.number}_run_{run_n}.csv', index=True)
        diff_neucf_results.append(ML_NEUCF_RESULTS)

        #### Evaluate Multiresolution Sampling ####
        if EVAL_MULTIRESOLUTION:

            synth_gen_set_t_random = sample_ddpm(N_USERS, diff_neucf, variational_ae, VAE_LATENT, noise_divider, 'random').cpu().numpy()
            upper_threshold = np.quantile(synth_gen_set_t_random.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
            lower_threshold = np.quantile(synth_gen_set_t_random.flatten(), 1 - SPARSITY)
            upper_equal_sparsity = pd.DataFrame((synth_gen_set_t_random >= upper_threshold).astype(int))
            lower_equal_sparsity = pd.DataFrame((synth_gen_set_t_random <= lower_threshold).astype(int))
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
            equal_sparsity = pd.concat([csr_equal_sparsity_zeros, csr_equal_sparsity_ones]).sample(frac=1, random_state=123)
            equal_sparsity_valid_train = pd.concat([equal_sparsity, row_valid_train]).sample(frac=1, random_state=123)  # Add the validation back into the synthetic
            equal_sparsity[0] = equal_sparsity[0] + (TRAIN_DATA.shape[0] + VALID_DATA.shape[0]) # Start the user column at the number of users in the training data
            # for idx, sampled_data in enumerate([equal_sparsity]):
            mutlires_diff_datasets = {
                # 'Original': TRAIN_PARTIAL_VALID_DATA,
                'Diff (Mutltiresolution)': equal_sparsity_valid_train,
                'Diff + Original (Multiresolution)': pd.concat([train_data, equal_sparsity], axis=0)
            }
            col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                         'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
            results = []
            for name, data in mutlires_diff_datasets.items():
                recall, ndcg = compute_neuralcf_results(data, valid_data, n_users=int(pd.DataFrame(data)[0].max())+1, n_items=int(pd.DataFrame(data)[1].max())+1)
                results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

            temp_df_t_random = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=mutlires_diff_datasets.keys())
            temp_df_t_random.to_csv(INTERMEDIATE_TRIAL_RESULTS + f'_multiresolution_diff_trial_{trial.number}_run_{run_n}.csv', index=True)
            diff_neucf_results_t_random.append(temp_df_t_random)

        ##################################
        ###### Evaluation MultiVAE #######
        ##################################
        if EVAL_MULTIVAE:
            raw_predictions = variational_ae.sample(N_USERS)
            upper_threshold = np.quantile(raw_predictions.flatten(),
                                          SPARSITY)  # Get the threshold for 0.1% sparsity
            lower_threshold = np.quantile(raw_predictions.flatten(), 1 - SPARSITY)
            upper_equal_sparsity = pd.DataFrame((raw_predictions >= upper_threshold).astype(int))
            lower_equal_sparsity = pd.DataFrame((raw_predictions <= lower_threshold).astype(int))
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
            equal_sparsity[0] = equal_sparsity[0] + (TRAIN_DATA.shape[0] + VALID_DATA.shape[
                0])  # Start the user column at the number of users in the training data
            # for idx, sampled_data in enumerate([equal_sparsity]):
            multivae_datasets = {
                # 'Original': TRAIN_PARTIAL_VALID_DATA,
                'MultiVAE': equal_sparsity_valid_train,
                'MultiVAE + Original': pd.concat([train_data, equal_sparsity], axis=0)
            }
            col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                         'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
            results = []
            for name, data in multivae_datasets.items():
                recall, ndcg = compute_neuralcf_results(data, valid_data, n_users=int(pd.DataFrame(data)[0].max()) + 1,
                                                        n_items=int(pd.DataFrame(data)[1].max()) + 1)
                results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

            ML_MLP_RESULTS = pd.DataFrame(np.concatenate(results, axis=1), index=col_names,
                                          columns=multivae_datasets.keys())
            ML_MLP_RESULTS.to_csv(INTERMEDIATE_TRIAL_RESULTS + f'_multivae_trial_{trial.number}_run_{run_n}.csv', index=True)
            multivae_neucf_results.append(ML_MLP_RESULTS)

    # Print Diffusion results
    diff_average_neucf_results = pd.DataFrame(np.round(np.mean([res.values for res in diff_neucf_results], axis=(0)), 4),
                                            index=col_names, columns=diff_datasets.keys())
    diff_max_neucf_results = pd.DataFrame(np.round(np.max([res.values for res in diff_neucf_results], axis=(0)), 4),
                                        index=col_names,
                                        columns=diff_datasets.keys())
    diff_std_neucf_results = pd.DataFrame(np.round(np.std([res.values for res in diff_neucf_results], axis=(0)), 4),
                                        index=col_names,
                                        columns=diff_datasets.keys())
    diff_average_neucf_results.to_csv(CSV_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
    diff_max_neucf_results.to_csv(CSV_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
    diff_std_neucf_results.to_csv(CSV_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)
    current_score = diff_average_neucf_results.loc[OPTIMIZATION_OBJECTIVE][
        1]  # 0 = Diff, 1 = Diff (0 Threshold), 2 = Diff (Equal Sparsity),3 = Diff + Original, 4 = Diff (0 Threshold) + Original, 5 = Diff (Equal Sparsity) + Original
    if current_score > BEST_SCORE:
        BEST_SCORE = current_score
        BEST_TRIAL = trial.number
    logger.info('Finished trial: ' + str(
        trial.number) + f'with score {current_score} and parameters {trial.params}. Best trial is {BEST_TRIAL} with score {BEST_SCORE}')
    logger.info(f'Time taken: {time.time() - start_time} seconds')

    logger.info(f'Mean\n{diff_average_neucf_results.to_markdown()}')
    logger.info(f'Max\n{diff_max_neucf_results.to_markdown()}')
    logger.info(f'Standard Dev\n{diff_std_neucf_results.to_markdown()}')

    ### Print MultiVAE results
    if EVAL_MULTIVAE:
        multivae_average_mlp_results = pd.DataFrame(
            np.round(np.mean([res.values for res in multivae_neucf_results], axis=(0)), 4),
            index=col_names, columns=multivae_datasets.keys())
        multivae_max_mlp_results = pd.DataFrame(
            np.round(np.max([res.values for res in multivae_neucf_results], axis=(0)), 4), index=col_names,
            columns=multivae_datasets.keys())
        multivae_std_mlp_results = pd.DataFrame(
            np.round(np.std([res.values for res in multivae_neucf_results], axis=(0)), 4), index=col_names,
            columns=multivae_datasets.keys())
        logger.info(f'Mean\n{multivae_average_mlp_results.to_markdown()}')
        logger.info(f'Max\n{multivae_max_mlp_results.to_markdown()}')
        logger.info(f'Standard Dev\n{multivae_std_mlp_results.to_markdown()}')
        multivae_average_mlp_results.to_csv(MULTIVAE_RESULTS_NAME + f'_mean_trial_{trial.number}.csv', index=True)
        multivae_max_mlp_results.to_csv(MULTIVAE_RESULTS_NAME + f'_max_trial_{trial.number}.csv', index=True)
        multivae_std_mlp_results.to_csv(MULTIVAE_RESULTS_NAME + f'_std_trial_{trial.number}.csv', index=True)

    # evaluate diffusion at different timesteps
    if EVAL_MULTIRESOLUTION:
        diff_neucf_results_t_random_avg = pd.DataFrame(
            np.round(np.mean([res.values for res in diff_neucf_results_t_random], axis=(0)), 4), index=col_names,
            columns=mutlires_diff_datasets.keys())
        diff_neucf_results_t_random_max = pd.DataFrame(
            np.round(np.max([res.values for res in diff_neucf_results_t_random], axis=(0)), 4), index=col_names,
            columns=mutlires_diff_datasets.keys())
        diff_neucf_results_t_random_std = pd.DataFrame(
            np.round(np.std([res.values for res in diff_neucf_results_t_random], axis=(0)), 4), index=col_names,
            columns=mutlires_diff_datasets.keys())
        logger.info(f'Random Timestep Mean: \n{diff_neucf_results_t_random_avg.to_markdown()}')
        logger.info(f'Random Timestep Max: \n{diff_neucf_results_t_random_max.to_markdown()}')
        logger.info(f'Random Timestep STD: \n{diff_neucf_results_t_random_std.to_markdown()}')
        diff_neucf_results_t_random_avg.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_t_random_mean_trial_{trial.number}.csv',
                                             index=True)
        diff_neucf_results_t_random_max.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_t_random_max_trial_{trial.number}.csv',
                                             index=True)
        diff_neucf_results_t_random_std.to_csv(MULTIRESOLUTION_RESULTS_NAME + f'_t_random_std_trial_{trial.number}.csv',
                                             index=True)
    return current_score  # Returns best average recall@50


def make_optuna_directories(root_dir, results_dir='nas_results'):
    # Getting current file path
    #dir_path = os.path.dirname(os.path.realpath(root_dir))
    # Create directories if they do not exist
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/logs')), exist_ok=True)
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/trial_results')), exist_ok=True)
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/vae_trial_results')), exist_ok=True)
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/study_results')), exist_ok=True)
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/study_db')), exist_ok=True)
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/multiresolution_trial_results')), exist_ok=True)
    os.makedirs(os.path.normpath(os.path.join(root_dir, f'./{results_dir}/intermediate_results')), exist_ok=True)


if __name__ == '__main__':

    # STATIC VARIABLES
    COLAB = False
    EVAL_MULTIVAE = True
    ALGORITHM_NAME = 'svd'  # 'SVD', 'MLP', 'NeuMF'
    DATASET_NAME = 'ml-100k'  # 'ADM', 'ML-1M', 'ML-100k', ALB,
    OPTIMIZATION_OBJECTIVE = 'Recall@10'
    OPTIMIZATION_SPARSITY = 'Equal+Original'  # 'Raw+Original', 'Zero+Original', 'Equal+Original', 'All+Original'
    EVAL_MULTIRESOLUTION = True
    RESULTS_PATH = '/content/drive/MyDrive/NAS_RESULTS' if COLAB else './temp'
    DATA_DIR_PATH = '/content/drive/MyDrive/data' if COLAB else './data'
    VAE_DIR_PATH = './temp_vae'

    os.mkdir(VAE_DIR_PATH) if not os.path.exists(VAE_DIR_PATH) else None
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    RESULT_DIR = f'nas_results_{DATASET_NAME}_{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}_{date_time}'
    make_optuna_directories(RESULTS_PATH, RESULT_DIR)
    assert os.path.exists(os.path.normpath(os.path.join(RESULTS_PATH, f"./{RESULT_DIR}")))
    print(f'Creating directories for results: {os.path.normpath(os.path.join(RESULTS_PATH, f"./{RESULT_DIR}"))}')
    print('Current date and time: ', date_time)

    LOGGER_FILE_NAME = f'{ALGORITHM_NAME}_log_{DATASET_NAME}_{date_time}.log'
    CSV_RESULTS_NAME = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/trial_results/{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}-results_{date_time}'))
    MULTIVAE_RESULTS_NAME = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/vae_trial_results/{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}-results_{date_time}'))
    MULTIRESOLUTION_RESULTS_NAME = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/multiresolution_trial_results/{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}-results_{date_time}'))
    OPTUNA_STUDY_NAME = f'{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@","_")}-{ALGORITHM_NAME}-study_{date_time}'
    OPTUNA_STUDY_PKL = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/study_results/{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}-study_{date_time}.pkl'))
    OPTUNA_DB_NAME = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/study_db/{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}-study_{date_time}.db'))
    INTERMEDIATE_TRIAL_RESULTS = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/intermediate_results/{DATASET_NAME}-{OPTIMIZATION_OBJECTIVE.replace("@", "_")}-{ALGORITHM_NAME}-results_{date_time}'))

    init_printout = f"NAS variables:" \
                    f"\n\tALGORITHM_NAME: {ALGORITHM_NAME}" \
                    f"\n\tDATASET_NAME: {DATASET_NAME}" \
                    f"\n\tOPTIMIZATION_OBJECTIVE: {OPTIMIZATION_OBJECTIVE}" \
                    f"\n\tLOGGER_FILE_NAME: {LOGGER_FILE_NAME}" \
                    f"\n\tCSV_RESULTS_NAME: {CSV_RESULTS_NAME}" \
                    f"\n\tOPTUNA_STUDY_NAME: {OPTUNA_STUDY_NAME}" \
                    f"\n\tOPTUNA_STUDY_PKL: {OPTUNA_STUDY_PKL}" \
                    f"\n\tOPTUNA_DB_NAME: {OPTUNA_DB_NAME}"
    print(init_printout)

    global logger
    # Getting current file path
    LOGGER_FILE_PATH = os.path.normpath(os.path.join(RESULTS_PATH, f'./{RESULT_DIR}/logs', LOGGER_FILE_NAME))
    print('Sending logger path to: ', LOGGER_FILE_PATH)
    # Create a custom logger
    logger = logging.getLogger(f'nas_logger')
    # Create handlers
    c_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler = logging.FileHandler(LOGGER_FILE_PATH)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)
    logger.info('Starting logger')
    logger.info(init_printout)
    #logger.info(f'Root Directory: {ROOT_DIR}')
    logger.info(f'Creating directories for results: {os.path.normpath(os.path.join(RESULTS_PATH, f"./{RESULT_DIR}"))}')
    logger.info(f'Moving results to {RESULTS_PATH} when runs are complete')
    logger.info(f'Current date and time: {date_time}')

    # Selecting datasets
    TRAIN_DATA, TRAIN_PARTIAL_VALID_DATA, VALID_DATA = load_data(dataset_name=DATASET_NAME.lower(), data_dir_path=DATA_DIR_PATH)
    N_ITEMS = TRAIN_DATA.shape[1]
    N_USERS = TRAIN_DATA.shape[0]
    SPARSITY = 1 - (TRAIN_DATA.nnz / (TRAIN_DATA.shape[0] * TRAIN_DATA.shape[1]))

    if ALGORITHM_NAME == 'SVD':
        func = lambda trial: mf_objective(trial, logger, BEST_SCORE)
    elif ALGORITHM_NAME == 'MLP':
        func = lambda trial: mlp_objective(trial, logger, BEST_SCORE)
    else:
        func = lambda trial: neucf_objective(trial, logger, BEST_SCORE)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                study_name=OPTUNA_STUDY_NAME, storage=f'sqlite:///{OPTUNA_DB_NAME}',
                                load_if_exists=True)

    study.enqueue_trial(   # Enqueue the default MultiVAE hyperparameters
        {'DIFF_TRAINING_EPOCHS': 200, 'DIFF_LR': 0.00001, 'N_HIDDEN_MLP_LAYERS': 4, 'TIMESTEPS': 100,
         'noise_divider': 1, 'VAE_LATENT': 200, 'VAE_HIDDEN': 600, 'VAE_LR': 0.001,
         'VAE_BATCH_SIZE': 500, 'BATCH_SIZE': 500}
    )
    study.optimize(func=func, n_trials=300)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")

    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
    logger.info(f'Train parameters: {trial}')
    joblib.dump(study, OPTUNA_STUDY_PKL)

    # Remove all the vae runs
    shutil.rmtree(VAE_DIR_PATH)

