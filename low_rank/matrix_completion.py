import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MatrixCompletion(nn.Module):
    def __init__(self, M_obs, rank, init='random'):
        super().__init__()
        if init == 'random':
            m, n = M_obs.shape
            U = torch.randn(m, rank, dtype=torch.float32)
            VT = torch.randn(n, rank, dtype=torch.float32).T
        elif init == 'svd':
            # U, S, V = np.linalg.svd(M_obs, full_matrices=False)
            U, S, VT = np.linalg.svd(M_obs)
            U = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
            VT = np.diag(np.sqrt(S[:rank])) @ VT[:rank, :]
            U = torch.tensor(U, dtype=torch.float32)
            VT = torch.tensor(VT, dtype=torch.float32)
        self.U = nn.Parameter(U)
        self.VT = nn.Parameter(VT)

        
    def forward(self):
        return self.U @ self.VT


def train_model(M_true, missing_rate, rank, lr=0.001, epochs=10000, momentum=0.9, init='random', repeat=1, lamb=0.1):
    """
    Args:
        M_true: full matrix (m x n) 
        missing rate: 
        rank: Target rank for completion
    """
    
    Ms_hat = []
    masks = []
    for _ in range(1, repeat + 1):
        print(f'# ============ [{_}-th] repeat =========== #')
        mask = get_missing_data(M_true, missing_rate)
        M_obs = M_true * mask
        masks.append(mask)
        model = MatrixCompletion(M_obs, rank, init=init)
        # optimizer = optim.SGD(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8000], gamma=0.5)
        
        criterion = nn.MSELoss()
        
        
        M_obs_t = torch.tensor(M_obs, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        for epoch in range(epochs):
            optimizer.zero_grad()
            M_pred = model()

            loss = criterion(M_pred[mask_t == 1], M_obs_t[mask_t == 1]) + lamb * (torch.norm(model.U) + torch.norm(model.VT))
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            scheduler.step()

        Ms_hat.append(model().detach().numpy())
    
    return Ms_hat, masks



def get_missing_data(data, missing_rate, mechanism='mcar'):

    while True:
        if mechanism == 'mcar':
            mask = 1 * (np.random.rand(*data.shape) > missing_rate) # False for missing value
        
        true_missing_rate = 1 - (np.count_nonzero(mask) / mask.size)
        if abs(missing_rate - true_missing_rate) < 0.01:
            break
    print(f'True missing rate: [{true_missing_rate:.4f}]')
    return mask


def process_data(data, threshold=1e-2):

    # complete the missing values with mean value
    mean_value = np.nanmean(np.where(data < threshold, np.nan, data), axis=1)

    # print(mean_value)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] < threshold:
                data[i, j] = mean_value[i]
    
    return data
