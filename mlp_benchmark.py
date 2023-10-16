import pandas as pd
import numpy as np
from scipy.sparse import vstack, csr_matrix
import pickle
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
import gc

import tensorflow as tf
#import tensorflow.keras.backend as k
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from collections import OrderedDict
from tqdm import tqdm

import os

# Local files
import utilities

### Configs
lr = 0.001
epochs = 200
batch_size = 16
num_factors = 8
dropout = 0.5
layers = [512, 512, 256, 256] #[64,32,16,8] # [512, 512, 256, 256] #
#layers = [64, 32, 17, 8]
reg_mf = 0
reg_layers = [0,0,0,0]
num_neg = 4

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(num_items,), name = 'user_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = initializers.HeNormal(),
                                  embeddings_regularizer = l2(reg_mf), input_length=1, mask_zero=True)(user_input)


    # MF part
    mlp_vector = Flatten()(MF_Embedding_User)
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)
        mlp_vector = Dropout(dropout)(mlp_vector)

    # Final prediction layer
    prediction = Dense(num_items, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(mlp_vector)

    model = Model(inputs=[user_input],
                  outputs=prediction)

    return model

def gpu_cleanup(objects):
    """Helps clean up things to prevent OOM during long training sessions"""
    if objects:
        del(objects)
    K.clear_session()
    gc.collect()


def compute_mlp_results(training_data, validation_data, combine_training=False):

    n_users = training_data.shape[0]
    n_items = training_data.shape[1]

    if type(training_data) is pd.DataFrame:
        training_data = training_data.values
    if type(training_data) is csr_matrix:
        training_data = training_data.toarray()

    recall_results = []
    ndcg_results = []

    model = get_model(num_users=n_users,
                      num_items=n_items,
                      mf_dim=num_factors,
                      layers=layers,
                      reg_layers=reg_layers,
                      reg_mf=reg_mf)

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10,
                                                        restore_best_weights=True,
                                                        monitor='val_root_mean_squared_error',
                                                        verbose=0,
                                                        mode='min',
                                                        min_delta=0.001)

    #train_train, train_test = utilities.split_train_test_proportion_from_csr_matrix(csr_matrix(combined_data), batch_size=10, random_seed=123)
    valid_train, valid_test = utilities.split_train_test_proportion_from_csr_matrix(validation_data, batch_size=10, random_seed=123)

    if combine_training:
        training_data = np.concatenate([training_data, valid_train.toarray()], axis=0)

    model.fit(x=training_data,  # Train on partial
              y=training_data, # Predict on all + new
              validation_split=0.2,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[early_stopping], verbose=0)

    predictions = model.predict(valid_train.toarray(), verbose=0)

    gpu_cleanup(model)

    masked_testing = utilities.mask_training_examples(sparse_training_set=valid_train, dense_matrix=predictions.copy())
    K_ = [1, 3, 5, 10, 20, 50]
    for k in K_:
        recall = utilities.recall_at_k_batch(masked_testing, valid_test, k=k)
        recall_results.append(np.round(np.nanmean(recall), 4))
    for k in K_:
        ndcg = utilities.NDCG_binary_at_k_batch(masked_testing, valid_test, k=k)
        ndcg_results.append(np.round(np.nanmean(ndcg), 4))
    return np.array(recall_results), np.array(ndcg_results)


def benchmark_datasets_mlp(MODEL_NAME, DATASET, include_original:bool=True):
    # Load original dataset
    print('Starting run for', MODEL_NAME, DATASET)

    root_dir = r'C:\Users\dlili\OneDrive\Documents\SJSU\Research\Experiments\WebConf\data'

    train_data = pickle.load(open(os.path.join(root_dir, f'./{DATASET}/{DATASET}_train_test.pkl'), 'rb'))
    valid_data = pickle.load(open(os.path.join(root_dir, f'./{DATASET}/{DATASET}_valid.pkl'), 'rb'))

    #FILE_NAME = f'{DATASET}-SVD-{MODEL_NAME}-results'

    ### Loading synthetic datasets
    testing_dataset = [np.load(os.path.join(root_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_1.npy')),
                       np.load(os.path.join(root_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_2.npy')),
                       np.load(os.path.join(root_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_3.npy')),
                       np.load(os.path.join(root_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_4.npy')),
                       np.load(os.path.join(root_dir, f'./{DATASET}/{MODEL_NAME}_{DATASET.upper()}_sample_5.npy')),]

    SPARSITY = 1 - (train_data.nnz / (train_data.shape[0] * train_data.shape[1]))

    MLP_RESULTS = []
    for testing_data in testing_dataset:
        if MODEL_NAME in ['CTGAN', 'TVAE']:
            testing_data[testing_data == -1] = 0
        threshold = np.quantile(testing_data.flatten(), SPARSITY)  # Get the threshold for 0.1% sparsity
        equal_sparsity = pd.DataFrame((testing_data >= threshold).astype(int))
        zero_threshold = pd.DataFrame((testing_data > 0).astype(int))

        datasets = {
            'Original': train_data,
            f'{MODEL_NAME}': testing_data,
            f'{MODEL_NAME} (0 Threshold)': zero_threshold,
            f'{MODEL_NAME} (Equal Sparsity)': equal_sparsity,
            f'{MODEL_NAME} + Original': np.concatenate([train_data.toarray(), testing_data], axis=0),
            f'{MODEL_NAME} (0 Threshold) + Original': np.concatenate([train_data.toarray(), zero_threshold], axis=0),
            f'{MODEL_NAME} (Equal Sparsity) + Original': np.concatenate([train_data.toarray(), equal_sparsity], axis=0)
        }

        if not include_original:
            del datasets['Original']
        if MODEL_NAME in ['CTGAN', 'TVAE']:
            del datasets[f'{MODEL_NAME} (0 Threshold)']
            del datasets[f'{MODEL_NAME} (Equal Sparsity)']
            del datasets[f'{MODEL_NAME} (0 Threshold) + Original']
            del datasets[f'{MODEL_NAME} (Equal Sparsity) + Original']

        col_names = ['Recall@1', 'Recall@3', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@50',
                     'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
        results = []
        for name, data in tqdm(datasets.items()):
            recall, ndcg = compute_mlp_results(data, valid_data, combine_training=True)
            results.append(np.concatenate([recall, ndcg]).reshape(-1, 1))

        ML_MLP_RESULTS = pd.DataFrame(np.concatenate(results, axis=1), index=col_names, columns=datasets.keys())
        #print(ML_MLP_RESULTS.to_markdown())
        MLP_RESULTS.append(ML_MLP_RESULTS)

    average_mlp_results = pd.DataFrame(np.round(np.mean([res.values for res in MLP_RESULTS], axis=(0)), 4), index=col_names, columns=datasets.keys())
    max_mlp_results = pd.DataFrame(np.round(np.max([res.values for res in MLP_RESULTS], axis=(0)), 4), index=col_names, columns=datasets.keys())
    std_mlp_results = pd.DataFrame(np.round(np.std([res.values for res in MLP_RESULTS], axis=(0)), 4), index=col_names, columns=datasets.keys())
    # average_mlp_results.to_csv(FILE_NAME+'_mean.csv', index=True)
    # max_mlp_results.to_csv(FILE_NAME+'_max.csv', index=True)
    # std_mlp_results.to_csv(FILE_NAME+'_std.csv', index=True)
    # print('\nMax\n', max_mlp_results.to_markdown(), sep='')
    # print('\nMean\n', average_mlp_results.to_markdown(), sep='')
    # print('\nStandard Deviation\n', std_mlp_results.to_markdown(), sep='')
    # print(f'Saved MLP results: {FILE_NAME}\n')
    return average_mlp_results, max_mlp_results, std_mlp_results



if __name__ == '__main__':

    avg_list = []
    max_list = []
    std_list = []

    models = ['CTGAN', 'TVAE', 'DIFFREC']

    DATASET = 'adm'
    for idx, MODEL in enumerate(models):
        average_svd_results, max_svd_results, std_svd_results = benchmark_datasets_mlp(MODEL, DATASET,
                                                                                       include_original=False )#rue if idx == 0 else False)
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


