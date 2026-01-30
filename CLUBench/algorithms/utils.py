
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import torch
import json

class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CustomDataset_idx(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx,self.data[idx], self.labels[idx]
    

class SimpleAE(nn.Module):
        
    def __init__(self, input_dim, hidden_dims=None):
        super(SimpleAE, self).__init__()

        if hidden_dims is None:
            self.hidden_dims = [int(input_dim / 2) if input_dim > 2 else 1, int(input_dim / 4) if input_dim > 4 else 1]
        else:
            self.hidden_dims = hidden_dims

        # encoder
        encoder_list = [
            ('encoder_fc1', nn.Linear(input_dim, self.hidden_dims[0]))
        ]

        for i in range(len(self.hidden_dims) - 1):
            encoder_list.append((f'encoder_act{i + 1}', nn.ReLU()))
            encoder_list.append((f'encoder_fc{i + 2}', nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])))
        
        self.encoder = nn.Sequential(OrderedDict(encoder_list))

        # decoder
        decoder_list = []
        for i in range(1, len(self.hidden_dims)):
            decoder_list.append((f'decoder_fc{len(self.hidden_dims) - i + 1}', nn.Linear(self.hidden_dims[-i], self.hidden_dims[-i-1])))
            decoder_list.append((f'decoder_act{len(self.hidden_dims) - i }', nn.ReLU()))
        decoder_list.append(('decoder_fc1', nn.Linear(self.hidden_dims[0], input_dim)))
        self.decoder = nn.Sequential(OrderedDict(decoder_list))


    def forward(self, x):
        x_hat = self.decoder(self.encoder(x))
        return x_hat
    

class TwoLayerDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, g1='relu', g2='relu', dropout_rate=0.2):
        super(TwoLayerDAE, self).__init__()
        self.dropout_rate = dropout_rate
        if g1 == 'relu':
            act = nn.ReLU
        else:
            act = nn.LeakyReLU
        
        if g2 == 'relu':
            act = nn.ReLU
        else:
            act = nn.LeakyReLU
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            act()
        )
    
    def forward(self, x):
        h = self.encoder(F.dropout(x, p=self.dropout_rate))
        x_hat = self.decoder(F.dropout(h, p=self.dropout_rate))
        return h, x_hat 


def clustetring_acc_hungarian(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    
    # Find optimal permutation using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Compute accuracy
    accuracy = cm[row_ind, col_ind].sum() / len(y_true)
    return accuracy


def compute_distance_matrix(X, metric, device):

    """
    Compute pairwise distance matrix for an n x d dataset.
    
    Args:
        X: Input data (numpy array or torch tensor) of shape (n_samples, n_features).
        metric: Distance metric ('euclidean', 'manhattan', 'cosine').
    
    Returns:
        Distance matrix of shape (n_samples, n_samples).
    """
    # Convert to PyTorch if not already
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    
    if metric == 'euclidean':
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        dist_sq = X_norm + X_norm.T - 2 * X @ X.T
        dist_sq = torch.clamp(dist_sq, min=0.0)  # Avoid numerical errors
        dist_matrix = torch.sqrt(dist_sq)
    
    elif metric == 'manhattan':
        # Sum of absolute differences
        dist_matrix = torch.cdist(X, X, p=1)
    
    elif metric == 'cosine':
        # 1 - (x_i • x_j) / (||x_i|| * ||x_j||)
        X_normalized = X / torch.norm(X, dim=1, keepdim=True)
        dist_matrix = 1 - X_normalized @ X_normalized.T
        dist_matrix = torch.clamp(dist_matrix, min=0.0)  # Avoid numerical errors
    
    else:
        raise ValueError(f"Unsupported metric: {self.metric}")
    dist_matrix = dist_matrix.numpy() if device == 'cpu' else dist_matrix.cpu().numpy()

    return dist_matrix


def json_load(path):
    
    with open(path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content
