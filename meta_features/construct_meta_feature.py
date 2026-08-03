__author__ = 'XF'
__date__ = '2026/01/06'


"""    Generate meta-features for clustering datasets.
"""
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from os import path as osp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import click

from meta_features.metaod import fix_nan
from CLUBench.configs import DATA_DIR, DATASETS
from utils import ROOT_DIR
from meta_features.metaod import generate_meta_features as metaod_features
from tools import obj_save, obj_load, new_dir
from iem_cuda import ClusteringMetricsCUDA


def kmeans_meta_features(data_name, device):

    meta_feature_dir = osp.join(ROOT_DIR, 'meta_features/kmeans')
    ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    kmeans_features = []

    data_path = osp.join(DATA_DIR, data_name)
    data = np.load(data_path, allow_pickle=True)
    X = data['x']
    X = np.array(X, dtype=np.float32)

    for k in ks:
        print(f'K:{k} =================================================')

        # For new datasets, you need to run kmeans to generate the predicted labels first, 
        # and then you can use the predicted labels to generate the meta-features.

        path = osp.join(meta_feature_dir, f'K={k}_predict_y.res')
        predict_ys = obj_load(path)
        predict_Y = np.array(predict_ys[data_name])

        assert len(X) == len(predict_Y)

        clum = ClusteringMetricsCUDA(device=device)
        results = clum.compute_all_metrics(X=X, labels=predict_Y)
        kmeans_features.extend(list(results.values()))
    print(f'The dimension of meta features from KMeans: {len(kmeans_features)}')

    return kmeans_features


@click.command()
@click.option('--data_begin', type=int, default=-1)
@click.option('--data_end', type=int, default=-1)
@click.option('--feature_type', type=str, default='stat', help='[stat, kmeans]')
@click.option('--device', type=str, default='cuda')


def main(data_begin, data_end, feature_type, device):

    if data_begin == -1:
        data_begin = 0
    if data_end == -1:
        data_end = len(DATASETS)

    if feature_type == 'stat':
        dim = 119
    elif feature_type == 'kmeans':
        dim = 130
    else:
        raise Exception(f'Unknown the feature type [{feature_type}].')
    valid_id = list(range(data_begin, data_end))
    meta_mat = np.zeros([len(valid_id), dim])

    for i, data_id in enumerate(valid_id):

        dataset_name = DATASETS[data_id]
        print(f'generating the meta-features for dataset:[{data_id + 1}] {dataset_name} =======================')

        if feature_type == 'stat':
            data_path = osp.join(DATA_DIR, dataset_name)
            data = np.load(data_path, allow_pickle=True)
            X = data['x']
            X = np.array(X, dtype=np.float32)

            meta_mat[i, :] = metaod_features(X)
        elif feature_type == 'kmeans':
            meta_mat[i, :] = kmeans_meta_features(data_name=dataset_name, device=device)

        indices = np.where(np.isinf(meta_mat[i:]))
        print(f'inf indices: [{indices}]')
        if len(indices) > 0:
            meta_mat[i, indices] = np.nan
        print(f'raw meta features:{len(meta_mat[i, :])}')
    
    meta_mat_transformed = meta_mat

    print(f'meta features:{meta_mat_transformed.shape}')
    save_dir = new_dir(ROOT_DIR, mk_dir=f'model_selection/meta_features/{feature_type}')
    save_path = osp.join(save_dir, f'{data_begin + 1}_{data_end}_meta_features_{str(meta_mat_transformed.shape[1])}.fea')
    obj_save(save_path, meta_mat_transformed)


if __name__ == '__main__':
    
    main()
