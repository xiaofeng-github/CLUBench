import torch
import numpy as np
from typing import Dict, Tuple
import warnings

class ClusteringMetricsCUDA:
    """CUDA-accelerated clustering metrics computation for high-dimensional data."""
    
    def __init__(self, device: str = 'cuda', dtype: torch.dtype = torch.float32):
        """
        Initialize CUDA metrics calculator.
        
        Args:
            device: 'cuda' or 'cpu'
            dtype: torch.float32 or torch.float64 for precision
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        print(f"Using device: {self.device}")
        
    def to_cuda(self, X: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to CUDA tensors."""
        X_tensor = torch.from_numpy(X).to(self.device, dtype=self.dtype)
        labels_tensor = torch.from_numpy(labels).to(self.device, dtype=torch.long)
        return X_tensor, labels_tensor
    
    def pairwise_distance_cuda(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        Efficient pairwise distance computation on CUDA.
        
        Uses matrix multiplication trick: ||x - y||² = ||x||² + ||y||² - 2x·y
        """
        if Y is None:
            Y = X
            
        X_norm = (X ** 2).sum(dim=1).view(-1, 1)
        Y_norm = (Y ** 2).sum(dim=1).view(1, -1)
        
        distances = X_norm + Y_norm - 2 * torch.mm(X, Y.t())
        # Ensure non-negative due to numerical precision
        distances = torch.clamp(distances, min=0.0)
        
        return torch.sqrt(distances + 1e-8)
    
    def compute_SC_cuda(self, X: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute Silhouette Coefficient (SC) statistics on CUDA.
        
        Optimized implementation using batched matrix operations.
        """
        n_samples = X.shape[0]
        n_clusters = labels.unique().shape[0]
        
        if n_clusters == 1 or n_clusters == n_samples:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        # Precompute pairwise distances in batches for memory efficiency
        silhouette_scores = torch.zeros(n_samples, device=self.device, dtype=self.dtype)
        
        # Process in batches to save memory
        batch_size = min(1000, n_samples)  # Adjust based on available GPU memory
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_indices = torch.arange(i, batch_end, device=self.device)
            
            # Compute distances for this batch
            X_batch = X[batch_indices]
            distances = self.pairwise_distance_cuda(X_batch, X)
            
            for idx, sample_idx in enumerate(batch_indices):
                sample_label = labels[sample_idx]
                
                # Intra-cluster distances (a)
                same_cluster_mask = labels == sample_label
                same_cluster_distances = distances[idx, same_cluster_mask]
                
                if same_cluster_mask.sum() > 1:
                    a = (same_cluster_distances.sum() - 0.0) / (same_cluster_mask.sum() - 1)  # Exclude self
                else:
                    a = torch.tensor(0.0, device=self.device)
                
                # Inter-cluster distances (b) - find minimum average distance to other clusters
                b = torch.tensor(float('inf'), device=self.device)
                
                for cluster_id in torch.unique(labels):
                    if cluster_id != sample_label:
                        other_cluster_mask = labels == cluster_id
                        if other_cluster_mask.sum() > 0:
                            avg_distance = distances[idx, other_cluster_mask].mean()
                            b = torch.min(b, avg_distance)
                
                # Compute silhouette for this sample
                if torch.isinf(b) or (a == 0 and b == 0):
                    silhouette_scores[sample_idx] = 0.0
                else:
                    max_val = torch.max(a, b)
                    silhouette_scores[sample_idx] = (b - a) / (max_val + 1e-10)
            
            # Clear cache to prevent memory issues
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Compute statistics
        valid_scores = silhouette_scores[~torch.isnan(silhouette_scores) & ~torch.isinf(silhouette_scores)]
        
        if len(valid_scores) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        sc_stats = {
            'mean': valid_scores.mean().item(),
            'std': valid_scores.std().item(),
            'min': valid_scores.min().item(),
            'max': valid_scores.max().item()
        }
        
        return sc_stats
    
    def compute_CH_cuda(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Calinski-Harabasz Index on CUDA.
        
        Uses efficient tensor operations without explicit pairwise distances.
        """
        n_samples = X.shape[0]
        n_clusters = labels.unique().shape[0]
        
        if n_clusters == 1 or n_clusters == n_samples:
            return 0.0
        
        # Overall mean
        overall_mean = X.mean(dim=0, keepdim=True)
        
        # Compute cluster statistics
        ssb = 0.0  # Between-cluster sum of squares
        ssw = 0.0  # Within-cluster sum of squares
        
        for cluster_id in torch.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum().item()
            
            if cluster_size > 0:
                cluster_data = X[cluster_mask]
                cluster_center = cluster_data.mean(dim=0, keepdim=True)
                
                # Between-cluster dispersion
                ssb += cluster_size * torch.sum((cluster_center - overall_mean) ** 2)
                
                # Within-cluster dispersion
                ssw += torch.sum((cluster_data - cluster_center) ** 2)
        
        if ssw == 0:
            return float('inf')
        
        ch_index = (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))
        
        return ch_index.item()
    
    def compute_DBI_cuda(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Davies-Bouldin Index on CUDA.
        
        Optimized with tensor operations and minimal memory usage.
        """
        n_clusters = labels.unique().shape[0]
        
        if n_clusters == 1:
            return 0.0
        
        # Compute cluster centers and scatter values
        cluster_centers = []
        cluster_scatters = []
        
        for cluster_id in torch.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_size = cluster_data.shape[0]
            
            if cluster_size > 0:
                cluster_center = cluster_data.mean(dim=0)
                cluster_centers.append(cluster_center)
                
                if cluster_size == 1:
                    cluster_scatters.append(torch.tensor(0.0, device=self.device))
                else:
                    # Intra-cluster scatter (average distance to center)
                    distances = torch.norm(cluster_data - cluster_center, dim=1)
                    cluster_scatters.append(distances.mean())
        
        if len(cluster_centers) < 2:
            return 0.0
        
        # Stack for vectorized operations
        cluster_centers = torch.stack(cluster_centers)
        cluster_scatters = torch.stack(cluster_scatters)
        n_valid_clusters = cluster_centers.shape[0]
        
        # Compute pairwise distances between centers
        # Use matrix multiplication trick for efficiency
        centers_norm = (cluster_centers ** 2).sum(dim=1).view(-1, 1)
        center_distances_sq = centers_norm + centers_norm.t() - 2 * torch.mm(cluster_centers, cluster_centers.t())
        center_distances_sq = torch.clamp(center_distances_sq, min=0.0)
        center_distances = torch.sqrt(center_distances_sq + 1e-8)
        
        # Mask self-distances
        mask = ~torch.eye(n_valid_clusters, device=self.device, dtype=torch.bool)
        center_distances_masked = center_distances * mask.float()
        center_distances_masked[~mask] = float('inf')
        
        # Compute R_ij = (S_i + S_j) / d_ij
        scatters_expanded_i = cluster_scatters.view(-1, 1).expand(-1, n_valid_clusters)
        scatters_expanded_j = cluster_scatters.view(1, -1).expand(n_valid_clusters, -1)
        R_matrix = (scatters_expanded_i + scatters_expanded_j) / center_distances_masked
        
        # For each cluster, find max R_ij (excluding self)
        max_R_values = R_matrix.max(dim=1).values
        
        # DBI is the average of these maxima
        # Filter out inf values
        valid_max_R = max_R_values[~torch.isinf(max_R_values)]
        
        if len(valid_max_R) == 0:
            return 0.0
        
        dbi = valid_max_R.mean().item()
        
        return dbi
    
    def compute_SSE_per_cluster_cuda(self, X: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute SSE statistics per cluster and overall.
        
        Returns:
            Dict with:
            - 'sse_total': Total SSE
            - 'sse_mean': Mean SSE per cluster
            - 'sse_std': Standard deviation of cluster SSEs
            - 'sse_max': Maximum cluster SSE
            - 'sse_min': Minimum cluster SSE
        """
        n_clusters = labels.unique().shape[0]
        
        if n_clusters == 0:
            return {
                'sse_total': 0.0,
                'sse_mean': 0.0,
                'sse_std': 0.0,
                'sse_max': 0.0,
                'sse_min': 0.0,
                'sse_per_cluster': []
            }
        
        unique_labels = torch.unique(labels)
        cluster_sse = []
        
        # Compute SSE for each cluster
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_size = cluster_data.shape[0]
            
            if cluster_size == 0:
                cluster_sse.append(0.0)
                continue
            
            # Compute centroid
            centroid = cluster_data.mean(dim=0)
            
            # Compute SSE for this cluster
            diff = cluster_data - centroid.unsqueeze(0)
            cluster_sse_val = torch.sum(diff ** 2).item()
            cluster_sse.append(cluster_sse_val)
        
        if not cluster_sse:
            return {
                'sse_total': 0.0,
                'sse_mean': 0.0,
                'sse_std': 0.0,
                'sse_max': 0.0,
                'sse_min': 0.0,
                'sse_per_cluster': []
            }
        
        cluster_sse_tensor = torch.tensor(cluster_sse, device=self.device)
        
        # Compute statistics
        sse_stats = {
            'sse_total': cluster_sse_tensor.sum().item(),
            'sse_mean': cluster_sse_tensor.mean().item(),
            'sse_std': cluster_sse_tensor.std().item(),
            'sse_max': cluster_sse_tensor.max().item(),
            'sse_min': cluster_sse_tensor.min().item(),
            'sse_per_cluster': cluster_sse
        }
        
        return sse_stats
    

    def compute_SSE_optimized_cuda(self, X: torch.Tensor, labels: torch.Tensor) -> float:
        """
        More optimized SSE computation using vectorized operations.
        
        This version is faster for moderate dimensions but uses more memory.
        """
        n_clusters = labels.unique().shape[0]
        
        if n_clusters == 0:
            return 0.0
        
        # Compute centroids for all clusters at once
        unique_labels = torch.unique(labels)
        centroids = []
        cluster_sizes = []
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_size = cluster_data.shape[0]
            
            if cluster_size > 0:
                centroids.append(cluster_data.mean(dim=0, keepdim=True))
                cluster_sizes.append(cluster_size)
        
        if not centroids:
            return 0.0
        
        # Stack centroids
        centroids = torch.cat(centroids, dim=0)  # Shape: (n_clusters, n_features)
        
        # For each sample, find distance to its assigned centroid
        sse = 0.0
        
        # Process samples in batches to save memory
        batch_size = min(5000, X.shape[0])
        
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            X_batch = X[i:end]
            labels_batch = labels[i:end]
            
            # Get centroids for this batch
            batch_centroids = centroids[labels_batch]  # Shape: (batch_size, n_features)
            
            # Compute squared distances
            diff = X_batch - batch_centroids
            sse += torch.sum(diff ** 2).item()
        
        return sse
    
    def compute_SSE_ratio_cuda(self, X: torch.Tensor, labels: torch.Tensor, current_sse: torch.Tensor) -> Dict[str, float]:
        """
        Compute SSE ratios useful for model selection:
        
        1. SSE_ratio_k: SSE(k) / SSE(k-1) (if comparing multiple k values)
        2. SSE_explained: 1 - (SSE / total_variance)
        3. Elbow_metric: Second derivative of SSE
        
        Note: For ratios, you need SSE for different k values.
        """
        # Compute total variance (SSE when k=1)
        overall_mean = X.mean(dim=0, keepdim=True)
        diff_total = X - overall_mean
        total_variance = torch.sum(diff_total ** 2).item()
        
        # Compute current SSE
        # current_sse = self.compute_SSE_optimized_cuda(X, labels)

        # print(current_sse, current_sse1)
        # exit(0)
        
        # Compute explained variance ratio (similar to R²)
        if total_variance > 0:
            explained_ratio = 1.0 - (current_sse / total_variance)
        else:
            explained_ratio = 0.0
        
        # If you have SSE for k-1, you can compute improvement ratio
        # Here we return the basic metrics
        
        return {
            'sse_current': current_sse,
            'total_variance': total_variance,
            'explained_ratio': explained_ratio,
            'unexplained_ratio': current_sse / total_variance if total_variance > 0 else 0.0
        }

    
    def compute_all_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute all three metrics in one pass with efficient memory usage.
        """
        # Convert to CUDA
        X_tensor, labels_tensor = self.to_cuda(X, labels)
        
        # Compute metrics
        sc_stats = self.compute_SC_cuda(X_tensor, labels_tensor)
        ch_score = self.compute_CH_cuda(X_tensor, labels_tensor)
        dbi_score = self.compute_DBI_cuda(X_tensor, labels_tensor)
        
        # Combine results
        results = {
            'silhouette_mean': sc_stats['mean'],
            'silhouette_std': sc_stats['std'],
            'silhouette_min': sc_stats['min'],
            'silhouette_max': sc_stats['max'],
            'calinski_harabasz': ch_score,
            'davies_bouldin': dbi_score,
        }

        # sse metric
        sse_stats = self.compute_SSE_per_cluster_cuda(X_tensor, labels_tensor)
        sse_ratios = self.compute_SSE_ratio_cuda(X_tensor, labels_tensor, sse_stats['sse_total'])

        results.update({
            'sse_total': sse_stats['sse_total'],
            'sse_mean': sse_stats['sse_mean'],
            'sse_std': sse_stats['sse_std'],
            'sse_max': sse_stats['sse_max'],
            'sse_min': sse_stats['sse_min'],
            'sse_explained_ratio': sse_ratios['explained_ratio'],
            'sse_unexplained_ratio': sse_ratios['unexplained_ratio']
        })
        
        return results
    


class BatchClusteringMetricsCUDA:
    """
    Optimized for processing 131 datasets in batches.
    Uses streaming computation to avoid GPU memory overflow.
    """
    
    def __init__(self, device: str = 'cuda', batch_size: int = 1000000):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.metrics_calculator = ClusteringMetricsCUDA(str(self.device))
        
    def process_datasets_batch(self, datasets: list) -> list:
        """
        Process multiple datasets in sequence, clearing memory between each.
        
        Args:
            datasets: List of tuples (X, labels) for each dataset
            
        Returns:
            List of metric dictionaries for each dataset
        """
        all_results = []
        
        for i, (X, labels) in enumerate(datasets):
            print(f"Processing dataset {i+1}/{len(datasets)}...")
            
            # Compute metrics
            results = self.metrics_calculator.compute_all_metrics(X, labels)
            all_results.append(results)
            
            # Clear GPU cache between datasets
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        return all_results