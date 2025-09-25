from .configs import DATA_DIR, DATASETS, osp

import numpy as np
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def load_data(data_name):

    assert data_name in DATASETS
    data_path = osp.join(DATA_DIR, data_name)
    data = np.load(data_path, allow_pickle=True)
    X, Y = data['x'], data['y']
    X = np.array(X, dtype=np.float32)

    return X, Y


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
