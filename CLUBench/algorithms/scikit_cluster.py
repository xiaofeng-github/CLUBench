from sklearn.cluster import KMeans as SK_KMeans
from sklearn.mixture import GaussianMixture as SK_GMM
from sklearn.cluster import DBSCAN as SK_DBSCAN
from sklearn.cluster import AgglomerativeClustering as SK_AC
from sklearn.cluster import Birch as SK_Birch
from sklearn.cluster import OPTICS as SK_OPTICS
from sklearn.cluster import SpectralClustering as SK_SC
from sklearn.cluster import MeanShift as SK_MeanShift
from sklearn.cluster import AffinityPropagation as SK_AffinityPropagation

from .base import BaseCluster
from .utils import compute_distance_matrix
import time
import torch
import numpy as np


class KMeans(BaseCluster):
    def __init__(self, n_clusters=8, **kwargs):
        super(KMeans, self).__init__()
        self.model = SK_KMeans(n_clusters=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class GMM(BaseCluster):
    def __init__(self, n_clusters=1, **kwargs):
        super(GMM, self).__init__()
        self.model = SK_GMM(n_components=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class DBSCAN(BaseCluster):
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        super(DBSCAN, self).__init__()
        if 'metric' in kwargs.keys() and kwargs['metric'] in ['euclidean', 'manhattan', 'cosine']:
            self.model = None
            self.kwargs = kwargs
            self.min_samples = min_samples
            self.eps = eps
        else:
            self.model = SK_DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    
    def fit_predict(self, X):

        if self.model is None:
            dist_matrix = compute_distance_matrix(X)
            eps_base = np.mean(dist_matrix[dist_matrix != 0])
            eps = self.eps * eps_base

            print(f'eps base: {eps_base:.4f}')
            print(f'eps: {eps:.4f}')

            self.kwargs['metric'] = 'precomputed'
        
            self.model = SK_DBSCAN(eps=eps, min_samples=self.min_samples, **self.kwargs)
            X = dist_matrix
        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class AggClu(BaseCluster):
    def __init__(self, n_clusters=2, **kwargs):
        super(AggClu, self).__init__()
        self.model = SK_AC(n_clusters=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class Birch(BaseCluster):
    def __init__(self, n_clusters=3, **kwargs):
        super(Birch, self).__init__()
        self.model = SK_Birch(n_clusters=n_clusters, **kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class OPTICS(BaseCluster):
    def __init__(self, min_samples=5, **kwargs):
        super(OPTICS, self).__init__()
        if 'metric' in kwargs.keys() and kwargs['metric'] in ['euclidean', 'manhattan', 'cosine']:
            self.model = None
            self.kwargs = kwargs
            self.min_samples = min_samples
        else:
            self.model = SK_OPTICS(min_samples=min_samples, **kwargs)
    
    def fit_predict(self, X):

        if self.model is None:
            dist_matrix = compute_distance_matrix(X)
            if 'max_eps' in self.kwargs.keys():
                eps_base = np.mean(dist_matrix[dist_matrix != 0])
                self.kwargs['max_eps'] = self.kwargs['max_eps'] * eps_base
                print(f'max_eps base: {eps_base:.4f}')
                print(f'max_eps: {self.kwargs["max_eps"]:.4f}')

            self.kwargs['metric'] = 'precomputed'
        
            self.model = SK_OPTICS(min_samples=self.min_samples, **self.kwargs)
            X = dist_matrix

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class SpeClu(BaseCluster):
    def __init__(self, n_clusters=8,  affinity='rbf', gamma=1, **kwargs):
        super(SpeClu, self).__init__()
        if affinity != 'rbf':
            self.model = SK_SC(n_clusters=n_clusters, affinity=affinity, **kwargs)
        else:
            self.model = None
            self.n_clusters = n_clusters
            self.gamma = gamma
    
    def rbf_affinity(self, X):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        return affinity_matrix
    
    def fit_predict(self, X):

        if self.model is None:

            # step 1: compute rbf kernel (affinity matrix)
            affinity_matrix = self.rbf_affinity(X)

            # step 2: compute nomalized Laplpcian
            D = torch.diag(torch.sum(affinity_matrix, dim=1))
            L = D - affinity_matrix
            L_norm = torch.inverse(D) @ L # normalized Laplacian: D^{-1}L

            # step 3: eigenvalue decomposition
            _, eigenvectors = torch.linalg.eigh(L_norm)

            # step 4: spectral embedding (top k eigenvectors)
            spectral_embedding = eigenvectors[:, :self.n_clusters].real

            spectral_embedding = spectral_embedding.numpy() if self.device == 'cpu' else spectral_embedding.cpu().numpy()
            # step 5: Kmeans clsutering
            km = KMeans(n_clusters=self.n_clusters)
            self.labels = km.fit_predict(spectral_embedding)
        else:
            self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class MeanShift(BaseCluster):
    def __init__(self, **kwargs):
        super(MeanShift, self).__init__()
        self.model = SK_MeanShift(**kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels


class AffinityPropagation(BaseCluster):
    def __init__(self, **kwargs):
        super(AffinityPropagation, self).__init__()
        self.model = SK_AffinityPropagation(**kwargs)
    
    def fit_predict(self, X):

        self.labels = self.model.fit_predict(X)
        self.times = time.time() - self.times
        return self.labels