import os.path as osp
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT_DIR = osp.dirname(osp.abspath(__file__))
ALL_PERFORMANCE_DIR = osp.join(ROOT_DIR, 'performance_matrix', 'all_hpcs')
BEST_PERFORMANCE_DIR = osp.join(ROOT_DIR, 'performance_matrix', 'best_hpc')

def obj_load(path):

    if osp.exists(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        raise OSError('no such path:%s' % path)


def fix_nan(X):

    col_mean = np.nanmean(X, axis = 0)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1]) 
    
    return X


def load_all_p():

    cluster = [
        'kmeans', 'kernel_kmeans', 'agglo', 'dbscan', 'birch', 'gmm', 'spectral_clustering','autosc', 
        'ssc', 'kfsc',  'kpc', 'meanshift', 's3comp', 'lrr', 'dec', 'idec',
        'dscn', 'pica', 'cc', 'edesc', 'dmicc', 'divc', 'p2ot', 'lfss'
    ]

    acc = []
    nmi = []
    ari = []
    all_num_p = 0
    for cm in cluster:
        path = osp.join(ALL_PERFORMANCE_DIR, f'{cm}.p')
        if osp.exists(path):
            print(f'cluster [{cm}] ==========================')
            p = obj_load(path)         
            print(f'num performance: [{len(p["acc"].values())}]')
            all_num_p += len(p['acc'].values())
            acc.extend(list(p['acc'].values()))
            nmi.extend(list(p['nmi'].values()))
            ari.extend(list(p['ari'].values()))

    print(f'num of all performance: [{all_num_p}]')
    return acc, nmi, ari


def load_best_p():

    cluster = [
        'kmeans', 'kernel_kmeans', 'agglo', 'dbscan', 'birch', 'gmm', 'spectral_clustering','autosc', 
        'ssc', 'kfsc',  'kpc', 'meanshift', 's3comp', 'lrr', 'dec', 'idec',
        'dscn', 'pica', 'cc', 'edesc', 'dmicc', 'divc', 'p2ot', 'lfss'
    ]
    acc = []
    nmi = []
    ari = []
    for cm in cluster:
        path = osp.join(BEST_PERFORMANCE_DIR, f'{cm}.p')
        if osp.exists(path):
            p = obj_load(path)
            acc.append(p['acc'])
            nmi.append(p['nmi'])
            ari.append(p['ari'])
    
    return acc, nmi, ari


def load_meta_features():

    meta_features_path = osp.join(ROOT_DIR, f'meta_features/meta_features_249_standardization.fea')

    meta_features = obj_load(meta_features_path)

    return meta_features