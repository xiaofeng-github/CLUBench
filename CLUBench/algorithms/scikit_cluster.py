import gc
from sklearn.cluster import KMeans as SK_KMeans
from sklearn.mixture import GaussianMixture as SK_GMM
from sklearn.cluster import DBSCAN as SK_DBSCAN
from sklearn.cluster import AgglomerativeClustering as SK_AC
from sklearn.cluster import Birch as SK_Birch
from sklearn.cluster import SpectralClustering as SK_SC
from sklearn.cluster import MeanShift as SK_MeanShift
from sklearn.cluster import estimate_bandwidth
from pyclustering.cluster.kmeans import kmeans as PyclusteringKMeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.utils.metric import distance_metric, type_metric


from .base import BaseCluster
from .utils import compute_distance_matrix
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

class KMeans(BaseCluster):

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, random_state=None, metric='euclidean', **kwargs):
        super(KMeans, self).__init__()

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.metric = metric

        self.sklearn_kmeans = SK_KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state, **kwargs)
    
    def fit_predict(self, X):

        if self.metric == 'euclidean':
            pred_y = self.sklearn_kmeans.fit_predict(X)

        elif self.metric == 'cosine':
            X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
            pred_y = self.sklearn_kmeans.fit_predict(X_normalized)

        elif self.metric == 'manhattan':
            metric = distance_metric(type_metric.MANHATTAN)
            list_X = X.tolist()

            if self.init == 'k-means++':
                initial_centers = kmeans_plusplus_initializer(list_X, self.n_clusters).initialize()
            elif self.init == 'random':
                initial_centers = random_center_initializer(list_X, self.n_clusters).initialize()

            self.pyclustering_kmeans = PyclusteringKMeans(list_X, initial_centers=initial_centers, metric=metric, itermax=self.max_iter)
            self.pyclustering_kmeans.process()
            clusters = self.pyclustering_kmeans.get_clusters()
            pred_y = np.empty(X.shape[0], dtype=int)
            for cluster_id, cluster in enumerate(clusters): 
                for index in cluster:
                    pred_y[index] = cluster_id
        
        self.labels = pred_y
        self.time = time.time() - self.time

        return self.labels


class GMM(BaseCluster):
    def __init__(self, n_clusters=1, **kwargs):
        super(GMM, self).__init__()
        self.model = SK_GMM(n_components=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.time = time.time() - self.time
        return self.labels


class DBSCAN(BaseCluster):
    def __init__(self, eps=0.5, min_samples=5, device=None, **kwargs):
        super(DBSCAN, self).__init__()
        kwargs.pop('n_clusters', None)
        if 'metric' in kwargs.keys() and kwargs['metric'] in ['euclidean', 'manhattan', 'cosine']:
            self.model = None
            self.kwargs = kwargs
            self.min_samples = min_samples
            if device is not None:
                self.device = device
            else:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.eps = eps
        else:
            self.model = SK_DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    
    def fit_predict(self, X):

        if self.model is None:
            dist_matrix = compute_distance_matrix(X, metric=self.kwargs['metric'], device=self.device)
            eps_base = np.mean(dist_matrix[dist_matrix != 0])
            eps = self.eps * eps_base

            print(f'eps base: {eps_base:.4f}')
            print(f'eps: {eps:.4f}')

            self.kwargs['metric'] = 'precomputed'
        
            self.model = SK_DBSCAN(eps=eps, min_samples=self.min_samples, **self.kwargs)
            X = dist_matrix
        self.labels = self.model.fit_predict(X)
        self.time = time.time() - self.time
        return self.labels


class AggClu(BaseCluster):
    def __init__(self, n_clusters=2, **kwargs):
        super(AggClu, self).__init__()
        self.model = SK_AC(n_clusters=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.time = time.time() - self.time
        return self.labels


class Birch(BaseCluster):
    def __init__(self, n_clusters=3, **kwargs):
        super(Birch, self).__init__()
        self.model = SK_Birch(n_clusters=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.time = time.time() - self.time
        return self.labels


class SpeClu(BaseCluster):
    def __init__(self, n_clusters, affinity='rbf', n_neighbors=10, gamma=1.0, random_state=None, device=None, **kwargs):
        super(SpeClu, self).__init__()

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    def distance_matrix(self, X):

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        dist_matrix = torch.cdist(X, X, p=2)
        dist_matrix = dist_matrix.cpu().data.numpy() if self.device == 'cuda' else dist_matrix.numpy()
        
        return dist_matrix

        
    
    def rbf_affinity(self, X):


        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Compute squared Euclidean distances
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        dist_sq = X_norm + X_norm.T - 2 * X @ X.T

        # Handle numerical stability (avoid negative distances due to floating-point errors)
        dist_sq = torch.clamp(dist_sq, min=0.0)

        gamma_base = 1 / (2 * torch.median(dist_sq[dist_sq != 0.0]))
        gamma = self.gamma * gamma_base
        
        # Compute RBF kernel
        affinity_matrix = torch.exp(-gamma * dist_sq)

        # affinity_matrix = affinity_matrix.numpy() if self.device == 'cpu' else affinity_matrix.cpu().numpy()
        
        return affinity_matrix
    
    def rbf_affinity_symmetric_batch(self, X, batch_size=2000):
        """
        Optimized version for symmetric matrix (saves 50% computation).
        """
        X = torch.tensor(X, dtype=torch.float16)
        n = X.shape[0]
        
        
        # Initialize only upper triangular will be computed
        square_dist = torch.zeros((n, n))
        
        n_batches = (n + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Computing symmetric RBF"):
            start_i = i * batch_size
            end_i = min((i + 1) * batch_size, n)
            batch_i = X[start_i:end_i].to(self.device)
            
            # Only compute for j >= i to avoid duplicate computation
            for j in range(i, n_batches):
                start_j = j * batch_size
                end_j = min((j + 1) * batch_size, n)
                batch_j = X[start_j:end_j].to(self.device)
                
                sq_dist_i = torch.cdist(batch_i, batch_j) ** 2
                
                square_dist[start_i:end_i, start_j:end_j] = sq_dist_i
                
                # Fill symmetric part (if not on diagonal)
                if i != j:
                    square_dist[start_j:end_j, start_i:end_i] = sq_dist_i.T

        gamma_base = 1 / (2 * torch.median(square_dist[square_dist != 0.0]))
        gamma = self.gamma * gamma_base
        
        # Compute RBF kernel
        affinity_matrix = torch.exp(-gamma * square_dist)
        
        
        return affinity_matrix

    def fit_predict(self, X):
        
        if self.affinity == 'rbf':
            if X.shape[0] < 20000:
                affinity_matrix = self.rbf_affinity(X)
                D = torch.diag(torch.sum(affinity_matrix, dim=1))

                # random walk normalized
                L = D - affinity_matrix
                L_norm = torch.inverse(D) @ L # normalized Laplacian: D^{-1}L
                # eigenvalue decomposition
                eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
 
                spectral_embedding = eigenvectors[:, 1:self.n_clusters + 1].real

                spectral_embedding = spectral_embedding.numpy() if self.device == 'cpu' else spectral_embedding.cpu().numpy()
            else:
                # compute rbf kernel (affinity matrix) in batches
                affinity_matrix = self.rbf_affinity_symmetric_batch(X)
       
                # compute nomalized Laplpcian
                D = torch.diag(torch.sum(affinity_matrix, dim=1))
                L = D - affinity_matrix
                
                # =====================
                del affinity_matrix
                gc.collect()
                # =====================

                L_norm = torch.inverse(D) @ L # normalized Laplacian: D^{-1}L
                L_norm = L_norm.numpy()

                # =====================
                del D
                del L
                gc.collect()
                # =====================
            
                print('Using scipy.sparse.linalg.eigsh for eigen decomposition.')
                eigenvalues, eigenvectors = eigsh(L_norm, k=self.n_clusters, which='SM')
                spectral_embedding = eigenvectors
                # =====================
                del L_norm
                gc.collect()
                # =====================

            # Kmeans clsutering
            km = KMeans(n_clusters=self.n_clusters)
            pred_y = km.fit_predict(spectral_embedding)
        elif self.affinity == 'nearest_neighbors':     
            c_matrix = self.distance_matrix(X)   
            sc = SK_SC(n_clusters=self.n_clusters,
                affinity='precomputed_nearest_neighbors',
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_jobs=-1,
                n_init=1,
                **self.kwargs
                )
            pred_y = sc.fit_predict(c_matrix)
        
        self.labels = pred_y
        self.time = time.time() - self.time
        return self.labels


class MeanShift(BaseCluster):

    def __init__(self, bandwidth=0.3, **kwargs):
        super(MeanShift, self).__init__()
        self.bandwidth = bandwidth
        self.kwargs = kwargs
        self.kwargs.pop('n_clusters', None)

    def estimate_bandwidth(self, X):

        self.bandwidth = estimate_bandwidth(X, quantile=self.bandwidth, n_jobs=-1)

    def fit_predict(self, X):

        self.estimate_bandwidth(X)
        meanshift = SK_MeanShift(bandwidth=self.bandwidth, **self.kwargs)

        self.labels = meanshift.fit_predict(X)
        self.time = time.time() - self.time

        return self.labels
