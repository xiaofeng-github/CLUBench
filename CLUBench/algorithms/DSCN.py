import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from sklearn.cluster import KMeans
from .base import BaseCluster
import time
from copy import deepcopy


def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y

def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp

def SP_for_affine_matrix(n_clusters,A):
    D = torch.diag(torch.sum(A, dim=1))
    D_inv_sqrt = torch.inverse(torch.sqrt(D))  # 使用torch.inverse代替torch.inv
    # 2. 计算对称归一化拉普拉斯矩阵
    L_sym = torch.eye(A.shape[0], device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    # 3. 特征分解
    # print("Has NaN:", torch.isnan(L_sym).any().item())
    # print("Has Inf:", torch.isinf(L_sym).any().item()) 
    #L_sym=L_sym.detach().cpu()
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)  # 使用更稳定的torch.linalg.eigh
    #print(eigenvalues)
    # 4. 处理特征向量（忽略接近0的特征值）
    tol = 1e-10  # 定义特征值为0的阈值
    nonzero_mask = eigenvalues > tol
    valid_eigenvectors = eigenvectors[:, nonzero_mask]
    # 确保n_clusters不超过有效特征向量数
    k = min(n_clusters, valid_eigenvectors.shape[1])
    U = eigenvectors[:, :n_clusters]
    # 5. 归一化行向量（使用torch.norm替代np.linalg.norm）
    U_normalized = U / (torch.norm(U, p=2, dim=1, keepdim=True)+1e-9)
    U_normalized[torch.isnan(U_normalized)] = 0  # 处理可能的NaN
    # 6. K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(U_normalized.cpu().numpy())
    labels = kmeans.labels_
    return labels

def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    #spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
    #                                      assign_labels='discretize')
    #spectral.fit(L)
    #grp = spectral.fit_predict(L)
    L = torch.from_numpy(L)
    grp = SP_for_affine_matrix(n_clusters=K, A=L)
    return grp, L


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class SimpleAE(nn.Module):
        
    def __init__(self, input_dim, hidden_dims=None,device='cuda'):
        super(SimpleAE, self).__init__()
        self.device=device

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
    
    def pretrain(self,x,epochs=500):
        train_data = DataLoader(dataset=x, batch_size=500, shuffle=True)
        #model = SimpleAE(input_dim=self.input_dim)
        epochs =epochs
        optimizer = Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        model = self.to(self.device)
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for batch in train_data:
                batch = batch.to(self.device)
                batch=batch.float()
                optimizer.zero_grad()
                x_hat = model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Pretrain Epoch {epoch}/{epochs}, Loss: {epoch_loss}")



class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)
        self.Coefficient.data.fill_diagonal_(0)

    def forward(self, x):  # shape=[n, d]
        self.Coefficient.data.fill_diagonal_(0)
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, feature_dim, hidden_dims,num_sample,device):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.feature_dim=feature_dim
        self.ae =SimpleAE(self.feature_dim,hidden_dims,device)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp

        return loss


def train(model,n_clusters,  # type: DSCNet
          x, y, epochs,stop_epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=10,pretrain_epochs=500):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.ae.pretrain(x,epochs=pretrain_epochs)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = n_clusters
    preds=[]
    for epoch in range(epochs):
        model.train()
        x=x.to(device)
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            print('Epoch %02d: loss=%.4f' %
                  (epoch, loss.item()))
        if (epoch+1)%stop_epochs==0 or epoch+1==epochs:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            preds.append(y_pred)
    return preds


class DSCN(BaseCluster):

    def __init__(self, n_clusters: int, hidden_dims=[512,128,256],stop_epochs=100,emb_dim=64,epochs: int = 300, lr: float = 1e-3, device: str = 'cuda',
                 weight_coef: float=1.0,weight_selfExp: float=150,alpha: float=0.04, dim_subspace: int=12, ro: float=8, show: int=10,pretrain_epochs=200,
                 final_epoch=True):
        super(DSCN, self).__init__()
        self.epochs = epochs
        self.hidden_dims = deepcopy(hidden_dims)
        self.hidden_dims.append(emb_dim)
        self.stop_epochs=stop_epochs
        self.emb_dim=emb_dim
        self.lr = lr
        self.weight_coef=weight_coef
        self.weight_selfExP=weight_selfExp
        self.alpha=alpha
        self.dim_subspace=dim_subspace
        self.ro=ro
        self.show=show
        self.device=device
        self.pretrain_epochs=pretrain_epochs
        self.n_clusters=n_clusters
        self.final_epoch = final_epoch
    

    def fit_predict(self, X):
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.input_dim = X.shape[1]
        self.Y = torch.ones(len(X))
        self.model = DSCNet(feature_dim=self.X.shape[1],hidden_dims=self.hidden_dims,num_sample=self.X.shape[0],device=self.device)
        self.model=self.model.to(self.device)
        self.labels=train(model=self.model,device=self.device,stop_epochs=self.stop_epochs,n_clusters=self.n_clusters,x=self.X,y=self.Y,epochs=self.epochs,lr=self.lr,weight_coef=self.weight_coef,weight_selfExp=self.weight_selfExP,alpha=self.alpha,dim_subspace=self.dim_subspace,ro=self.ro,show=self.show,pretrain_epochs=self.pretrain_epochs)
        if self.final_epoch:
            self.labels=self.labels[-1]
        self.time=time.time()-self.time
        return self.labels
