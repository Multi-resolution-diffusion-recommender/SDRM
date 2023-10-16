import math

import pandas as pd
import numpy as np
from scipy.sparse import vstack, csr_matrix, coo_matrix
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, recall_score
from collections import OrderedDict
import gc

# import tensorflow.keras.backend as k
import torch.optim as optim
from collections import OrderedDict
from tqdm import tqdm
import time
DEVICE = 'cuda'

import utilities

import os
from os.path import normpath

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### Configs
lr = 0.001
epochs = 20
batch_size = 256
num_factors = 32
dropout = 0.5
#layers = [64, 32, 16, 8]
reg_mf = 0
reg_layers = [0, 0, 0, 0]
num_neg = 1

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        """
		user_num: number of users
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                          self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def compute_neuralcf_results(training_data, validation_data, n_users, n_items, verbose=False, eval_recall=True):

    K_ = [1, 3, 5, 10, 20, 50]
    neuralcf_results = 'neuralcf_temp'
    os.mkdir(neuralcf_results) if not os.path.exists(neuralcf_results) else None

    recall_results = []
    ndcg_results = []
    model_type = 'NeuMF'

    model = NCF(n_users, n_items, factor_num=8, num_layers=3,
                      dropout=0.5, model='NeuMF-end', GMF_model=None, MLP_model=None)
    model.to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    ########################### TRAINING #####################################
    best_loss, best_epoch, early_stop_count, best_recall = math.inf, 0, 0, 0
    training_loss = []
    validation_loss = []
    for epoch in range(epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()

        # Split the training data into two sets
        train_data, valid_data = train_test_split(training_data, test_size=0.2, shuffle=True)
        n_postitive_samples = train_data[train_data[2] == 1].shape[0]
        negative_dataset = train_data[train_data[2] == 0].sample(n=n_postitive_samples * num_neg, replace=True)
        # combine the positive and negative samples
        train_data = pd.concat([train_data, negative_dataset]).sample(frac=1)

        for start_idx in range(0, train_data.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, train_data.shape[0])

            X = torch.tensor(train_data[start_idx:end_idx].to_numpy(), dtype=torch.int32, device=DEVICE)

            # user = X[:, 0].to(torch.int64)
            # item = X[:, 1].to(torch.int64)
            # label = X[:, 2]

            model.zero_grad()
            prediction = model(X[:, 0], X[:, 1])
            loss = loss_function(prediction, X[:, 2].float())
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        ### Evaluate the model
        model.eval()
        with torch.no_grad():
            X_valid = torch.tensor(valid_data.to_numpy(), dtype=torch.int32, device=DEVICE)
            # valid_user = X_valid[:, 0].to(torch.int64)
            # valid_item = X_valid[:, 1].to(torch.int64)
            # valid_label = X_valid[:, 2]
            valid_prediction = model(X_valid[:, 0], X_valid[:, 1])  #.detach().cpu().numpy()
            valid_loss = loss_function(valid_prediction, X_valid[:, 2].float())
            validation_loss.append(valid_loss.item())

            if eval_recall:

                valid_user_list = valid_data[0].unique()
                valid_item_list = training_data[1].unique()
                # valid_user_item = np.array(np.meshgrid(valid_user_list, valid_item_list), dtype=np.int32).T.reshape(-1,
                #                                                                                                     2)  # Create a meshgrid of all user-item pairs
                valid_user_item = torch.cartesian_prod(torch.tensor(valid_user_list, device=DEVICE, dtype=torch.int32),
                                                       torch.tensor(valid_item_list, device=DEVICE, dtype=torch.int32))
                # torch.meshgrid(torch.tensor(valid_user_list, device=DEVICE, dtype=torch.int32),
                #                torch.tensor(valid_item_list, device=DEVICE, dtype=torch.int32))
                # Batch testing
                valid_batch_size = 10000
                batch_pred = []
                for i in range(0, valid_user_item.shape[0], valid_batch_size):
                    start_idx = i
                    end_idx = min(i + valid_batch_size, valid_user_item.shape[0])
                    #valid_user_item_batch = pd.DataFrame(valid_user_item[start_idx:end_idx], columns=[0, 1])
                    valid_prediction_batch = model(valid_user_item[start_idx:end_idx][:,0],
                                                   valid_user_item[start_idx:end_idx][:,1])
                    # valid_prediction_batch = model(X_test[:, 0], X_test[:, 1])
                    batch_pred.extend(valid_prediction_batch.detach().cpu().numpy().flatten().tolist())
                valid_user_item = pd.DataFrame(valid_user_item.detach().cpu().numpy(), columns=[0, 1])
                valid_user_item['predictions'] = batch_pred
                valid_user_item['validation_labels'] = valid_user_item.merge(validation_data, on=[0, 1], how='left')[2].fillna(
                    0)
                # Mask training examples from predictions
                valid_user_item['training_labels'] = valid_user_item.merge(training_data, on=[0, 1], how='left')[2]
                valid_user_item['masked_predictions'] = valid_user_item['predictions'].where(valid_user_item['training_labels'].isnull(), other=-np.inf)

                test_actual, _, _, _, _ = utilities.create_csr_from_df(valid_user_item, user_col=0, item_col=1,
                                                                       rating_col='validation_labels')
                test_pred, _, _, _, _ = utilities.create_csr_from_df(valid_user_item, user_col=0, item_col=1,
                                                                     rating_col='masked_predictions')
                recall = np.nanmean(utilities.recall_at_k_batch(test_pred.toarray(), test_actual, k=10))
                if recall > best_recall:
                    best_recall = recall
                    best_recall_epoch = epoch
                    best_loss = valid_loss.item()
                    torch.save(model.state_dict(), f'./{neuralcf_results}/{model_type}_{epoch}.pth')
                else:
                    early_stop_count += 1

                if early_stop_count == 10:
                    break

                elapsed_time = time.time() - start_time
                if verbose:
                    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
                    print("Training loss: {:.3f}\tValidation loss: {:.3f}\tRecall: {:.4f}".format(np.mean(training_loss), np.mean(validation_loss), recall))

            # Stop training if validation loss does not improve for 10 epochs
            else:
                # Check for early stopping
                if best_loss > valid_loss.item():
                    best_loss = valid_loss.item()
                    best_epoch = epoch
                    early_stop_count = 0
                    torch.save(model.state_dict(), f'./{neuralcf_results}/{model_type}_{epoch}.pth')
                else:
                    early_stop_count += 1

                if early_stop_count == 10:
                    break

                elapsed_time = time.time() - start_time
                if verbose:
                    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
                    print("Training loss: {:.3f}\tValidation loss: {:.3f}".format(np.mean(training_loss), np.mean(validation_loss)))

        if eval_recall:
            if verbose:
                print("End. Best epoch {:03d}: best_loss = {:.4f}, best_recall = {:.4f}".format(best_epoch, best_loss, best_recall))
        else:
            if verbose:
                print("End. Best epoch {:03d}: best_loss = {:.4f}".format(best_epoch, best_loss))

    # Load best model
    model.load_state_dict(torch.load(f'./{neuralcf_results}/{model_type}_{best_epoch}.pth'))
    model.eval()

    ########################### TESTING #####################################
    with torch.no_grad():
        # predict for every item for every user
        valid_user_list = validation_data[0].unique()
        valid_item_list = training_data[1].unique()
        #valid_user_item = np.array(np.meshgrid(valid_user_list, valid_item_list), dtype=np.int32).T.reshape(-1, 2)  # Create a meshgrid of all user-item pairs
        valid_user_item = torch.cartesian_prod(torch.tensor(valid_user_list, device=DEVICE, dtype=torch.int32),
                                               torch.tensor(valid_item_list, device=DEVICE, dtype=torch.int32))
        testing_batch_size = 10000
        batch_pred = []
        for i in range(0, valid_user_item.shape[0], testing_batch_size):
            start_idx = i
            end_idx = min(i + testing_batch_size, valid_user_item.shape[0])
            #valid_user_item_batch = pd.DataFrame(valid_user_item[start_idx:end_idx], columns=[0, 1])
            #X_test = torch.tensor(valid_user_item_batch.to_numpy(), dtype=torch.int32, device=DEVICE)
            valid_prediction_batch = model(valid_user_item[start_idx:end_idx][:,0],
                                           valid_user_item[start_idx:end_idx][:,1])
            batch_pred.extend(valid_prediction_batch.detach().cpu().numpy().flatten().tolist())
        #X_test = torch.tensor(valid_user_item.to_numpy(), dtype=torch.int32, device=DEVICE)
        #X_test = torch.meshgrid(valid_user_list, valid_item_list, indexing='ij', dtype=torch.int32)
        #test_prediction = model(X_test[:, 0], X_test[:, 1])
        valid_user_item = pd.DataFrame(valid_user_item.detach().cpu().numpy(), columns=[0, 1])
        valid_user_item['predictions'] = batch_pred
        valid_user_item['validation_labels'] = valid_user_item.merge(validation_data, on=[0, 1], how='left')[2].fillna(0)
        # Mask training examples from predictions
        valid_user_item['training_labels'] = valid_user_item.merge(training_data, on=[0, 1], how='left')[2]
        valid_user_item['masked_predictions'] = valid_user_item['predictions'].where(valid_user_item['training_labels'].isnull(), other=-np.inf)

        # test_actual = csr_matrix((valid_user_item['validation_labels'], (valid_user_item[0], valid_user_item[1])),
        #                          shape=(valid_user_item[0].nunique(), valid_user_item[1].nunique()))
        # test_pred = csr_matrix((valid_user_item['masked_predictions'], (valid_user_item[0], valid_user_item[1])),
        #                          shape=(valid_user_item[0].nunique(), valid_user_item[1].nunique()))
        test_actual, _, _, _, _ = utilities.create_csr_from_df(valid_user_item, user_col=0, item_col=1, rating_col='validation_labels')
        test_pred, _, _, _, _ = utilities.create_csr_from_df(valid_user_item, user_col=0, item_col=1, rating_col='masked_predictions')

        for k in K_:
            ndcg = utilities.NDCG_binary_at_k_batch(test_pred.toarray(), test_actual, k=k)
            ndcg_results.append(np.round(np.nanmean(ndcg), 4))
            recall = utilities.recall_at_k_batch(test_pred.toarray(), test_actual, k=k)
            recall_results.append(np.round(np.nanmean(recall), 4))

    return np.array(recall_results), np.array(ndcg_results)


def benchmark_datasets_neuralcf(MODEL_NAME, DATASET, include_original: bool = True, data_dir='./data', eval_recall: bool = True):
    # Load original dataset
    print('Starting run for', MODEL_NAME, DATASET)

    # Loading training data and converting it into user-item-rating format
    _train_data = pickle.load(open(os.path.join(data_dir, f'./{DATASET}/{DATASET}_train_test.pkl'), 'rb'))
    row_train_data = pd.DataFrame(
        [_train_data.tocoo().row, _train_data.tocoo().col, _train_data.tocoo().data]).T.sort_values(by=0)

    # Loading Validation data and converting it into user-item-rating format
    _valid_data = pickle.load(open(os.path.join(data_dir, f'./{DATASET}/{DATASET}_valid.pkl'), 'rb'))
    row_valid_data = pd.DataFrame(
        [_valid_data.tocoo().row, _valid_data.tocoo().col, _valid_data.tocoo().data]).T.sort_values(by=0)
    row_valid_data[0] += _train_data.shape[0]  # Start the user column at the number of users in the training data

    # Splitting the validation data into two sets and ignoring the zeros (same splits as in MLP/SVD)
    valid_train, valid_test = utilities.split_train_test_proportion_from_csr_matrix(_valid_data, batch_size=10,
                                                                                    random_seed=123, ignore_zeros=True)
    # Converting the validation dataset(s) into user-item-rating format
    row_valid_train_no_zeros = pd.DataFrame(
        [valid_train.tocoo().row, valid_train.tocoo().col, valid_train.tocoo().data]).T.sort_values(by=0)
    row_valid_train_no_zeros[0] += _train_data.shape[0]
    # Start the user column at the number of users in the training data

    row_valid_test_no_zeros = pd.DataFrame(
        [valid_test.tocoo().row, valid_test.tocoo().col, valid_test.tocoo().data]).T.sort_values(by=0)
    row_valid_test_no_zeros[0] += _train_data.shape[0]
    # Start the user column at the number of users in the training data

    # Get only the zero ratings from the validation dataset
    row_valid_data_only_zeros = row_valid_data[row_valid_data[2] == 0].sample(frac=1, random_state=123)
    # Combine the zero ratings with training and testing data
    row_valid_train = pd.concat(
        [row_valid_data_only_zeros[:int(row_valid_data_only_zeros.shape[0] / 2)], row_valid_train_no_zeros])
    valid_data = pd.concat(
        [row_valid_data_only_zeros[int(row_valid_data_only_zeros.shape[0] / 2):], row_valid_test_no_zeros]).sample(
        frac=1, random_state=123)
    # Combine valid train with train data
    train_data = pd.concat([row_train_data, row_valid_train]).sample(frac=1, random_state=123)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)

    train_data = train_data[~train_data.isin(valid_data)].dropna()
    train_data.reset_index(drop=True, inplace=True)

    ### Loading synthetic datasets
    testing_dataset = [np.load(normpath(os.path.join(data_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_1.npy'))),
                       np.load(normpath(os.path.join(data_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_2.npy'))),
                       np.load(normpath(os.path.join(data_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_3.npy'))),
                       np.load(normpath(os.path.join(data_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_4.npy'))),
                       np.load(normpath(os.path.join(data_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_5.npy')))]

    SPARSITY = 1 - (_train_data.nnz / (_train_data.shape[0] * _train_data.shape[1]))

    MLP_RESULTS = []
    for testing_data in testing_dataset:
        if MODEL_NAME in ['MultiVAE', 'CODIGEM', 'DIFFREC']:
            upper_threshold = np.quantile(testing_data.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
            lower_threshold = np.quantile(testing_data.flatten(), 1 - SPARSITY)
            upper_equal_sparsity = pd.DataFrame((testing_data >= upper_threshold).astype(int))
            lower_equal_sparsity = pd.DataFrame((testing_data <= lower_threshold).astype(int))
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
            equal_sparsity[0] = equal_sparsity[0] + (_train_data.shape[0] +_valid_data.shape[0]) # Start the user column at the number of users in the training data
            # equal_sparsity  # Start the user column at the number of users in the training data

            datasets = {
                'Original': train_data,
                f'{MODEL_NAME}': equal_sparsity_valid_train,
                f'{MODEL_NAME} + Original': pd.concat([train_data, equal_sparsity], axis=0),
            }
        if MODEL_NAME in ['TVAE', 'CTGAN']:  # TODO eventually figure out where there are extra people in the datasets not do max
            testing_data = pd.DataFrame(testing_data)
            # Drop rows with all zeros
            testing_data = testing_data[(testing_data.T != 0).any()]
            # Replace -1 with 0
            #testing_data = testing_data.replace(-1, 0)
            csr_testing_data = csr_matrix(testing_data)
            equal_sparsity = pd.DataFrame(
                [csr_testing_data.tocoo().row, csr_testing_data.tocoo().col,
                 csr_testing_data.tocoo().data]).T.sort_values(by=0)
            equal_sparsity[equal_sparsity[2] == -1] = 0

            csr_testing_data_ones_valid_train = pd.concat([equal_sparsity, row_valid_train]).sample(frac=1, random_state=123)  # Add the validation back into the synthetic
            equal_sparsity[0] = equal_sparsity[0] + (_train_data.shape[0] +_valid_data.shape[0]) # Start the user column at the number of users in the training data
            # equal_sparsity  # Start the user column at the number of users in the training data

            datasets = {
                'Original': train_data,
                f'{MODEL_NAME}': csr_testing_data_ones_valid_train,
                f'{MODEL_NAME} + Original': pd.concat([train_data, equal_sparsity], axis=0),
            }

        if not include_original:
            del datasets['Original']

        col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                     'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
        results = []
        for name, data in tqdm(datasets.items()):
            recall, ndcg = compute_neuralcf_results(pd.DataFrame(data), valid_data, n_items=int(pd.DataFrame(data)[1].max())+1,  # TODO eventually figure out where there are extra people in the datasets not do max
                                                    n_users=int(pd.DataFrame(data)[0].max())+1, eval_recall=eval_recall)
            results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

        ML_MLP_RESULTS = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=datasets.keys())
        # print(ML_MLP_RESULTS.to_markdown())
        MLP_RESULTS.append(ML_MLP_RESULTS)

    average_mlp_results = pd.DataFrame(np.round(np.mean([res.values for res in MLP_RESULTS], axis=(0)), 4),
                                       index=col_names, columns=datasets.keys())
    max_mlp_results = pd.DataFrame(np.round(np.max([res.values for res in MLP_RESULTS], axis=(0)), 4), index=col_names,
                                   columns=datasets.keys())
    std_mlp_results = pd.DataFrame(np.round(np.std([res.values for res in MLP_RESULTS], axis=(0)), 4), index=col_names,
                                   columns=datasets.keys())
    # average_mlp_results.to_csv(FILE_NAME+'_mean.csv', index=True)
    # max_mlp_results.to_csv(FILE_NAME+'_max.csv', index=True)
    # std_mlp_results.to_csv(FILE_NAME+'_std.csv', index=True)
    # print('\nMax\n', max_mlp_results.to_markdown(), sep='')
    # print('\nMean\n', average_mlp_results.to_markdown(), sep='')
    # print('\nStandard Deviation\n', std_mlp_results.to_markdown(), sep='')
    # print(f'Saved MLP results: {FILE_NAME}\n')
    return average_mlp_results, max_mlp_results, std_mlp_results


if __name__ == "__main__":
    print("Running NeuralCF benchmark...")

    root_dir = r'C:\Users\dlili\OneDrive\Documents\SJSU\Research\Experiments\WebConf\data'

    # Load data
    avg_list = []
    max_list = []
    std_list = []

    models = ['CTGAN', 'TVAE', 'MultiVAE', 'CODIGEM', 'DIFFREC']

    DATASET = 'ml-1m'
    for idx, MODEL in enumerate(models):
        average_svd_results, max_svd_results, std_svd_results = benchmark_datasets_neuralcf(MODEL, DATASET,
                                                                                            include_original=True if idx == 0 else False,
                                                                                            data_dir=root_dir, eval_recall=False)
        print('\nMean\n', average_svd_results.to_markdown(), sep='')
        print('\nMax\n', max_svd_results.to_markdown(), sep='')
        print('\nStandard Deviation\n', std_svd_results.to_markdown(), sep='')
        avg_list.append(average_svd_results)
        max_list.append(max_svd_results)
        std_list.append(std_svd_results)

    # Combine all results
    avg_df = pd.concat(avg_list, axis=1)
    max_df = pd.concat(max_list, axis=1)
    std_df = pd.concat(std_list, axis=1)
    print('\nMean\n', avg_df.to_markdown(), sep='')
    print('\nMax\n', max_df.to_markdown(), sep='')
    print('\nStandard Deviation\n', std_df.to_markdown(), sep='')
