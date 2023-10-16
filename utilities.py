from collections import defaultdict
from scipy.sparse import csr_matrix, vstack
import bottleneck as bn
import numpy as np
import pandas as pd
import math

np.seterr(invalid='ignore')

# Convert to utility matrix
def create_csr_from_df(df, user_col, item_col, rating_col):
    """
    CODE FROM: https://www.jillcates.com/pydata-workshop/html/tutorial.html

    Generates a compressed sparse matrix from ratings dataframe.

    Args:
        df: pandas dataframe containing 3 columns (user_id, song_id, rating)
        user_col: str, name of column from df that contains the user id's
        item_col: str, name of column from df that contains the item id's
        rating_col: str, name of column from df that contains the ratings

    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        item_mapper: dict that maps item id's to item indices
        item_inv_mapper: dict that maps item indices to item id's
    """
    M = df[user_col].nunique()
    N = df[item_col].nunique()

    user_mapper = dict(zip(np.unique(df[user_col]), list(range(M))))
    item_mapper = dict(zip(np.unique(df[item_col]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df[user_col])))
    item_inv_mapper = dict(zip(list(range(N)), np.unique(df[item_col])))

    user_index = [user_mapper[i] for i in df[user_col]]
    item_index = [item_mapper[i] for i in df[item_col]]

    X = csr_matrix((df[rating_col], (user_index, item_index)), shape=(M, N))

    return X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper


def precision_recall_at_k(predictions, k=10, threshold=0.5, input_kind='suprise'):
    """
    Computes precision and recall @ k for each user when given user-item-ratings format.

    Parameters
    ----------
    predictions: Surprise Dataset or pd.DataFrame
        Predictions made from surprise dataset object or pandas dataframe
    k: int
        Number of top predictions to evaluate
    threshold: float
        Threshold for valid prediction or not
    input_kind: str
        Determines if predictions is a Surprise Data object or pandas dataframe

    Returns
    -------
        tuple(dict[user_id] = precision, dict[user_id] = recall)
            Dictionary with precision and recall for each user

    Note
    ----
    If predicting with custom input (not suprise predictions),
    data must be in the format of a 3 column pandas df or np.array
    where each sample structure is:

    [user_id, item, rating]

    Code adapted from: https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
    """

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    if input_kind == 'suprise':
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))
    else:  # If predicting with custom input
        for uid, true_r, est in predictions:
            user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))  # Item is recommended and actually recommended
            for (est, true_r) in user_ratings[:k]  # For top k recommended items
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def mask_training_examples(sparse_training_set, dense_matrix):
    """Excludes the training examples from being tested"""
    dense_matrix[
        sparse_training_set.nonzero()] = -np.inf  # Sets the already rated values to negative inf to prevent ranking ordering
    return dense_matrix


def NDCG_binary_at_k_batch(X_pred: np.ndarray, heldout_batch: csr_matrix, k=100):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance

    Function from: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])

    NDCG = (DCG / IDCG)
    return NDCG


def recall_at_k_batch(X_pred, heldout_batch, k=100):
    """
    Code from Function from: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
    :param X_pred:
    :param heldout_batch:
    :param k:
    :return:
    """
    batch_users = X_pred.shape[0]  # Get number of users

    idx = bn.argpartition(-X_pred, k, axis=1)  # Get the indexes of the top k predictions
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)  # Create a zero array of the same shape as X_pred
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True  # Set the top k predictions to True

    if isinstance(heldout_batch, np.ndarray):
        X_true_binary = heldout_batch > 0
    else:
        X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)  # counts how many are actually true and predicted true
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))  # Computes recall by dividing true positives by total positives (for each user)

    return recall


def split_train_test_proportion_from_csr_matrix(csr_data, test_prop=0.2, batch_size=None, random_seed=None, ignore_zeros=False):
    """
    Splits a csr_matrix into a training and test portion where both the training and test will have the same rows and
    columns.

    Parameters
    ----------
    csr_data: scipy.sparse.csr_matrix
        sparse input data matrix
    test_prop: float, default = 0.2
        Portion of each row to split
    batch_size: int, default = None
        Number of rows to process at a time. Helpful if the input data is very large
    random_seed: int, default = None
        Random seed initialization
    ignore_zeros: bool, default = False
        If True, ignores all zero values in the input matrix

    Returns
    -------
    train_csr_matrix, test_csr_matrix
    """
    if random_seed:
        np.random.seed(random_seed)
    if type(csr_data) is not csr_matrix:
        raise TypeError('Input data is not of type csr_matrix')
    train_list, test_list = list(), list()
    train_csr_list, test_csr_list = list(), list()
    batch_counter = 0

    # Removes all entries with 0 in csr matrix
    if ignore_zeros:
        csr_data.eliminate_zeros()

    for row in csr_data:
        n_items = row.indices.shape[0]  # Get how many items the user has rated
        if n_items < 2:
            print(f'Warning: skipping user with {n_items} items rated')
            continue
        idx = np.zeros(n_items, dtype='bool')  # Create a zero-array to represent the indexes of items the user has rated
        idx[np.random.choice(n_items, size=math.ceil(test_prop * n_items), replace=False).astype('int32')] = True  # randomly select test_prop items to be testing data
        train_item_vector = np.zeros(row.shape[1])
        test_item_vector = np.zeros(row.shape[1])  # Empty 0 vector to represent items
        np.put(train_item_vector, ind=row.indices[~idx], v=1)  # Replaces 0's with values at training data indexes (inplace)
        np.put(test_item_vector, ind=row.indices[idx], v=1)  # Replaces 0's with values at test data indexes (inplace)
        train_list.append(train_item_vector)
        test_list.append(test_item_vector)
        batch_counter += 1
        if batch_size and batch_counter >= batch_size:  # If batch
            train_csr_list.append(csr_matrix(np.array(train_list)))  # Compress training list
            test_csr_list.append(csr_matrix(np.array(test_list)))  # Compress testing list
            train_list, test_list = list(), list()  # Reset the training and test lists
            batch_counter = 0  # Reset batch size counter

    if train_list:  # Checks for edge case of training list being empty
        train_csr_list.append(csr_matrix(np.array(train_list)))  # Compress training list
    if test_list:  # Checks for edge case of testing list being empty
        test_csr_list.append(csr_matrix(np.array(test_list)))  # Compress testing list
    train_csr_matrix = vstack(train_csr_list)
    test_csr_matrix = vstack(test_csr_list)

    return train_csr_matrix, test_csr_matrix


if __name__ == '__main__':
    import pickle

    amazon_data = pickle.load(open('./amazon_music_crs_matrix.pkl', 'rb'))
    train_csr_matrix, test_csr_matrix = split_train_test_proportion_from_csr_matrix(amazon_data, batch_size=5)
    pass