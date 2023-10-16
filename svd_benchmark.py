import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from collections import OrderedDict
import tqdm
import pickle
import os
from scipy.sparse import vstack, csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
import warnings
import utilities

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compute_mf_results(training_dataset: csr_matrix, testing_dataset: csr_matrix, synthetic_data: csr_matrix=None, nnmf: bool=False, only_synthetic: bool=False):
    """
    Compute the results for the matrix factorization

    Takes in a training dataset, testing dataset, and synthetic dataset and computes the recall and ndcg results on
    20 % of the testing dataset. The synthetic dataset is used to augment the training dataset.

    :param training_dataset:
    :param testing_dataset:
    :param synthetic_data:
    :param nnmf:
    :return:
    """
    # Splitting the testing data into a subset for testing and validation
    test_data, valid_data = utilities.split_train_test_proportion_from_csr_matrix(testing_dataset, batch_size=1000, random_seed=123)
    if only_synthetic:
        if not type(synthetic_data) is np.ndarray:
            synthetic_data = np.array(synthetic_data)
        training_data = np.concatenate([synthetic_data, test_data.toarray()], axis=0)
    else:
        training_data = np.concatenate([training_dataset.toarray(), test_data.toarray()], axis=0)  # Combine training and testing data portion
    recall_results = []
    ndcg_results = []

    if not only_synthetic or synthetic_data is None:
        # If synethetic data is not None, then we are using the synthetic data to augment the training data
        if not type(synthetic_data) is np.ndarray:
            synthetic_data = np.array(synthetic_data)
        combined_data = np.concatenate([training_data, synthetic_data], axis=0)
    else:
        combined_data = np.concatenate([training_data], axis=0)

    if nnmf:
        MF = NMF(n_components=15, max_iter=50)
    else:
        MF = TruncatedSVD(n_components=20, n_iter=100)  # TruncatedSVD(n_components=100, n_iter=20)
    Z = MF.fit_transform(combined_data)
    new_X = MF.inverse_transform(Z)

    # Mask the training examples
    masked_testing = utilities.mask_training_examples(sparse_training_set=training_data,
                                                          dense_matrix=new_X[:training_data.shape[0]].copy())
    K_ = [1, 3, 5, 10, 20, 50]
    for k in K_:
        if only_synthetic:
            recall = utilities.recall_at_k_batch(masked_testing[synthetic_data.shape[0]:synthetic_data.shape[0]+valid_data.shape[0]], valid_data, k=k)
            ndcg = utilities.NDCG_binary_at_k_batch(masked_testing[synthetic_data.shape[0]:synthetic_data.shape[0]+valid_data.shape[0]], valid_data, k=k)
        else:
            recall = utilities.recall_at_k_batch(masked_testing[training_dataset.shape[0]: training_dataset.shape[0]+valid_data.shape[0]], valid_data, k=k)
            ndcg = utilities.NDCG_binary_at_k_batch(masked_testing[training_dataset.shape[0]: training_dataset.shape[0]+valid_data.shape[0]], valid_data, k=k)
        ndcg_results.append(np.round(np.nanmean(ndcg), 4))
        recall_results.append(np.round(np.nanmean(recall), 4))

    return np.array(recall_results), np.array(ndcg_results)


def benchmark_datasets(MODEL_NAME, DATASET, include_original: bool=True, only_synthetic: bool=False, data_dir_path='./data'):
    # Load original dataset
    print('Starting run for', MODEL_NAME, DATASET)

    train_data = pickle.load(open(os.path.join(data_dir_path, f'./{DATASET}/{DATASET}_train_test.pkl'), 'rb'))
    valid_data = pickle.load(open(os.path.join(data_dir_path, f'./{DATASET}/{DATASET}_valid.pkl'), 'rb'))

    #FILE_NAME = f'{DATASET}-SVD-{MODEL_NAME}-results'

    ### Loading synthetic datasets
    testing_dataset = [np.load(os.path.join(data_dir_path, f'{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_1.npy')),
                       np.load(os.path.join(data_dir_path, f'{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_2.npy')),
                       np.load(os.path.join(data_dir_path, f'{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_3.npy')),
                       np.load(os.path.join(data_dir_path, f'{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_4.npy')),
                       np.load(os.path.join(data_dir_path, f'{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_5.npy'))]

    SPARSITY = 1 - (train_data.nnz / (train_data.shape[0] * train_data.shape[1]))

    total_results = []


    for testing_data in testing_dataset:
        if MODEL_NAME in ['CTGAN', 'TVAE']:
            testing_data[testing_data == -1] = 0
        threshold = np.quantile(testing_data.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
        equal_sparsity = pd.DataFrame((testing_data >= threshold).astype(int))
        zero_threshold = pd.DataFrame((testing_data > 0).astype(int))

        datasets = OrderedDict({
            'Original': None,
            f'{MODEL_NAME} Raw Logits': testing_data,
            f'{MODEL_NAME} Zero Threshold': zero_threshold,
            f'{MODEL_NAME} Equal Sparsity': equal_sparsity,
        })

        if not include_original:
            del datasets['Original']
        if MODEL_NAME in ['CTGAN', 'TVAE']:
            datasets[f'{MODEL_NAME}'] = datasets[f'{MODEL_NAME} Raw Logits']
            del datasets[f'{MODEL_NAME} Raw Logits']
            del datasets[f'{MODEL_NAME} Zero Threshold']
            del datasets[f'{MODEL_NAME} Equal Sparsity']

        col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                     'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']

        results = []
        for name, data in tqdm.tqdm(datasets.items()):
            recall, ndcg = compute_mf_results(train_data, valid_data, synthetic_data=data, nnmf=False, only_synthetic=only_synthetic)
            results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

        total_results.append(pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=datasets.keys()))

    average_svd_results = pd.DataFrame(np.round(np.mean([res.values for res in total_results], axis=(0)), 4), index=col_names, columns=datasets.keys())
    max_svd_results = pd.DataFrame(np.round(np.max([res.values for res in total_results], axis=(0)), 4), index=col_names, columns=datasets.keys())
    std_svd_results = pd.DataFrame(np.round(np.std([res.values for res in total_results], axis=(0)), 4), index=col_names, columns=datasets.keys())
    # max_svd_results.to_csv(FILE_NAME+'_max.csv', index=True)
    # average_svd_results.to_csv(FILE_NAME+'_mean.csv', index=True)
    # std_svd_results.to_csv(FILE_NAME+'_std.csv', index=True)
    # print('\nMean\n', average_svd_results.to_markdown(), sep='')
    # print('\nMax\n', max_svd_results.to_markdown(), sep='')
    # print('\nStandard Deviation\n', std_svd_results.to_markdown(), sep='')

    return average_svd_results, max_svd_results, std_svd_results


if __name__ == '__main__':
    # Change current directory to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    avg_list = []
    max_list = []
    std_list = []

    models = ['MultiVAE']
    data_dir_path = './data'

    DATASET = 'alb'
    for idx, MODEL in enumerate(models):
        average_svd_results, max_svd_results, std_svd_results = benchmark_datasets(MODEL, DATASET,
                                                                                   include_original=False,
                                                                                   only_synthetic=False,
                                                                                   data_dir_path=data_dir_path)
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
