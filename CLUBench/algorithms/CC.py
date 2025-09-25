import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import normalize
import time
from .base import BaseCluster

class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims,):
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
        layers.append(nn.Linear(prev_dim, prev_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # 展平输入（如果是图像等）
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class Network(nn.Module):
    def __init__(self, input_dim, hidden_dims,class_num,emb_dim,device):
        super(Network, self).__init__()
        self.hidden_dims = hidden_dims
        self.feature_dim = input_dim
        self.emb_dim=emb_dim
        self.cluster_num = class_num
        self.encoder = SimpleMLP(input_dim,hidden_dims)
        #nn.init.xavier_uniform_(self.encoder.weight)
        self.device=device
        self.instance_projector = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], self.emb_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], self.cluster_num),
            nn.Softmax(dim=1)
        )
        self=self.to(self.device)

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.encoder(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    

class InstanceLoss(nn.Module):
    def __init__(self,temperature, device):
        super(InstanceLoss, self).__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.mask = self.mask_correlated_samples(z_i.shape[0])
        N = 2 *z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, z_i.shape[0])
        sim_j_i = torch.diag(sim, -z_i.shape[0])

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

#-------------------------------


def train(model, dataset,epochs, stop_epochs,noise_std=0.1,lr=1e-3,batch_size=256,instance_temp=0.5,cluster_temp=1,device='cuda'):
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_instance = InstanceLoss(instance_temp, device).to(
        device)
    criterion_cluster = ClusterLoss(model.cluster_num, cluster_temp, device).to(device)
    preds=[]
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
            x_i=batch.float()
            noise = torch.randn_like(x_i) * noise_std
            x_j=x_i+noise
                #print(batch.shape)
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            loss_instance = criterion_instance(z_i, z_j)
            loss_cluster = criterion_cluster(c_i, c_j)
            loss = loss_instance + loss_cluster
            #print(loss)
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % ( 0.0),
                loss="%.8f" % float(loss.item()),
                dlb="%.4f" % (0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1)%stop_epochs==0 or (epoch+1)==epochs:
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
            x=model.forward_cluster(x)
            if y_hat==None:
                y_hat=x
            else:
                y_hat=torch.concat([y_hat,x])
        return y_hat.detach().cpu()
            


class ConClu(BaseCluster):

    def __init__(self, n_clusters:int, hidden_dims: list = [512, 256, 128], stop_epochs: int = 50,
                 emb_dim: int = 64, epochs: int = 300, lr: float = 1e-3, 
                 device: str = 'cuda',
                 instance_temp:float = 0.5,
                 cluster_temp: float = 1,
                 noise_std: float=0.1,
                 batch_size:int=256,
                 final_epoch=True):
        super(ConClu, self).__init__()
        
        self.epochs = epochs
        self.hidden_dims=hidden_dims
        self.stop_epochs=stop_epochs
        self.emb_dim=emb_dim
        self.lr = lr
        self.final_epoch = final_epoch
       
        self.noise_std=noise_std
        self.batch_size=batch_size
        self.device=device
        self.instance_temp=instance_temp
        self.cluster_temp=cluster_temp
        self.n_clusters=n_clusters

    def fit_predict(self, X):

        self.X = X
        self.input_dim = self.X.shape[1]
        self.Y = np.ones(self.X.shape[0])
        self.dataset=CustomDataset(self.X,self.Y)
        self.net=Network(self.X.shape[1],
                         self.hidden_dims, self.n_clusters,self.emb_dim,device=self.device).to(self.device)

        self.labels =  train(self.net,self.dataset,epochs=100,stop_epochs=self.stop_epochs,lr=1e-4,batch_size=256,device=self.device,instance_temp=self.instance_temp,cluster_temp=self.cluster_temp)
        
        if self.final_epoch:
            self.labels = self.labels[-1]
        self.times = time.time() - self.times
        return self.labels
