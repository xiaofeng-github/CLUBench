import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from .base import BaseCluster
from tqdm import tqdm
import time


class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        简化版的多层感知机
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如 [512, 256, 128]
            output_dim: 输出维度
        """
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.softmax(self.network(x),dim=1)
    

class PUILoss(nn.Module):

    def __init__(self, lamda=2.0,device='cuda'):
        super(PUILoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda
        self.device=device
    def forward(self, x, y):
        """Partition Uncertainty Index
        
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]
        
        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == y.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        #pui=F.cosine_similarity(x.t(),y.t())
        #print(pui.shape)
        #print(x)
        #print(x)
        #print(x.shape)
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        #print(pui.sum())
        loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).to(self.device))
        #print(loss_ce)
        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * torch.log(p)).sum()
        #print(p)
        #print(loss_ne)
        #print(loss_ne)
        #print(loss_ce)

        return loss_ce + self.lamda * loss_ne

#-------------------------------
# In PICAnet, the only augementation way is to add gaussian noise to original data. 
class PICAnet(nn.Module):
    def __init__(self, feature_dim, hidden_dims,num_classes,lamda,noise_std,device):
        super(PICAnet, self).__init__()
        self.feature_dim=feature_dim
        self.num_classes=num_classes
        self.net =SimpleMLP(self.feature_dim,hidden_dims,self.num_classes)
        self.lamda=lamda
        self.noise_std=noise_std
        self.device=device
        self.PUILoss=PUILoss(lamda=self.lamda,device=self.device)
        self=self.to(self.device)
        #print(self.parameters)

    def forward(self, x):  # shape=[n, c, w, h]
        noise = torch.randn_like(x) * self.noise_std
        noisy_data = x+ noise
        #print(x)
        z = self.net(x)
        z_noise=self.net(noisy_data)
        #print(z_noise)
        loss=self.PUILoss(z,z_noise)
        return loss
    def predcit(self,x):
        z=self.net(x)
        return z


def train(model, dataset,epochs,stop_epochs,lr=1e-3,batch_size=256,device='cuda'):
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    preds=[]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "lss": "%.8f" % 0.0,
            },
            disable=False
        )
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                batch
            ) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
                
            batch = batch.to(device)
            batch=batch.float()
                #print(batch.shape)
            loss = model(batch)
            #print(loss)
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % ( 0.0),
                loss="%.8f" % float(loss.item()),
                dlb="%.4f" % (0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
        if (epoch+1)%stop_epochs==0 or epoch+1==epochs:
            preds.append(test(model,dataset,batch_size))
    return preds

def test(model,dataset,batch_size):
        model.eval()
        test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
        y_hat=None
        for data in test_dataloader:
            x,y=data
            x=x.to(model.device)
            x=x.float()
            x=model.predcit(x)
            x=torch.argmax(x, dim=1)
            if y_hat==None:
                y_hat=x
            else:
                y_hat=torch.concat([y_hat,x])
        return y_hat.detach().cpu().numpy()

class PICA(BaseCluster):

    def __init__(self, n_clusters:int, hidden_dims=[512,256,128],stop_epochs=50,
                 epochs: int = 300, lr: float = 1e-3, device: str = 'cuda',
                 lamda: float=0.1, noise_std: float=0.1,batch_size:int=256,
                 final_epoch=True):
        super(PICA, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.lamda=lamda
        self.noise_std=noise_std
        self.batch_size=batch_size
        self.device=device
        self.n_clusters=n_clusters
        self.hidden_dims=hidden_dims
        self.stop_epochs=stop_epochs
        self.final_epoch = final_epoch

    def fit_predict(self, X):
        self.dataset=CustomDataset(X, np.ones(X.shape[0]))
        model= PICAnet(feature_dim=X.shape[1],hidden_dims=self.hidden_dims,num_classes=self.n_clusters,lamda=self.lamda,noise_std=self.noise_std,device=self.device)
        self.labels=train(model,self.dataset,epochs=self.epochs,stop_epochs=self.stop_epochs,lr=self.lr,batch_size=self.batch_size,device=self.device)
        if self.final_epoch:
            self.labels=self.labels[-1]
        self.time = time.time() - self.time
        return self.labels