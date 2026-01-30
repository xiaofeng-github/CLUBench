from .configs import DATA_DIR, HPC_DIR, DATASETS, osp

import json
import numpy as np
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def load_data(data_name, nomalization='z-score'):

    assert data_name in DATASETS
    data_path = osp.join(DATA_DIR, data_name)
    data = np.load(data_path, allow_pickle=True)
    X, Y = data['x'], data['y']
    X = np.array(X, dtype=np.float32)
    X = _normalization(X, method=nomalization)

    return X, Y


def _normalization(X, method=None):

    if method == 'z-score':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / (std + 1e-6)
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val)
    elif method is None:
        X_normalized = X
    else:
        raise ValueError(f'Unknown normalization method: {method}')
    
    return X_normalized


def load_hpc(hpc_name):

    hpc_path = osp.join(HPC_DIR, f'{hpc_name.lower()}.json')
    hpc = json_load(hpc_path)
    return hpc


def json_load(path):
    
    with open(path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def clustering_evaluation(y_true, y_predict):

    acc = float(clustetring_acc_hungarian(y_true, y_predict))
    nmi = float(normalized_mutual_info_score(y_true, y_predict))
    ari = float(adjusted_rand_score(y_true, y_predict))

    return {'acc': acc, 'nmi': nmi, 'ari': ari}


def clustetring_acc_hungarian(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    
    # Find optimal permutation using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Compute accuracy
    accuracy = cm[row_ind, col_ind].sum() / len(y_true)
    return accuracy
