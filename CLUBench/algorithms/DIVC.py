import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
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

class DivClustLoss(torch.nn.Module):

    def __init__(self, threshold=1., NMI_target=1., NMI_interval=20, threshold_rate=0.99, divclust_mbank_size=10000):
        super(DivClustLoss, self).__init__()
        self.threshold = threshold
        self.NMI_target = NMI_target
        self.NMI_interval = NMI_interval
        self.threshold_rate = threshold_rate
        self.current_threshold = threshold
        self.divclust_mbank_size = divclust_mbank_size
        self.memory_bank = None

    def loss(self, assignments, threshold):
        if not isinstance(assignments, torch.Tensor):
            assignments = torch.stack(assignments)
        K, N, C = assignments.shape
        id_rem = F.one_hot(torch.arange(K, device=assignments.device), K).bool()
        clustering_similarities = torch.einsum("qbc,kbd->qkcd", assignments, assignments).permute(1, 0, 2, 3)[
            ~id_rem].view(K * (K - 1), C, C)

        clustering_sim_aggr = clustering_similarities.max(-1)[0].mean(-1)
        loss = F.relu(clustering_sim_aggr - threshold).sum()

        return loss

    def forward(self, assignments: torch.Tensor, step=None):
        if isinstance(assignments, torch.Tensor):
            if len(assignments.shape) == 2:
                assignments = assignments.unsqueeze(0)
        clusterings = len(assignments)

        if clusterings == 1 or self.NMI_target == 1:
            return torch.tensor(0., device=assignments.device, requires_grad=True), self.threshold, assignments

        if self.NMI_target == 1:
            threshold = self.get_adaptive_threshold(threshold, self.adaptive_threshold, step)
        else:
            self.update_mb(assignments)
            threshold = self.get_NMI_threshold(self.NMI_target, step)
        self.current_threshold = threshold

        if isinstance(assignments, torch.Tensor):
            assignmentsl2 = F.normalize(assignments, p=2, dim=1)
        else:
            assignmentsl2 = [F.normalize(assignments_k, p=2, dim=0) for assignments_k in assignments]

        if threshold == 1.:
            return torch.tensor(0., device=assignments.device, requires_grad=True), threshold, assignments

        loss = self.loss(assignmentsl2, self.current_threshold)
        return loss, threshold, assignments

    @torch.no_grad()
    def update_mb(self, assignments):
        labels = assignments.argmax(-1)
        if self.memory_bank is None:
            self.memory_bank = labels.cpu().numpy()
        else:
            self.memory_bank = np.concatenate([labels.cpu().numpy(), self.memory_bank], axis=1)
        self.memory_bank = self.memory_bank[:, :self.divclust_mbank_size]

    def get_NMI_threshold(self, NMI_target, step):
        threshold = self.current_threshold
        if step is None or step % self.NMI_interval == 0:
            k = self.memory_bank.shape[0]
            NMIs = []
            for k1 in range(k):
                for k2 in range(k1 + 1, k):
                    NMIs.append(normalized_mutual_info_score(self.memory_bank[k1], self.memory_bank[k2]))
            NMI = np.mean(NMIs)
            if NMI > NMI_target:
                threshold = self.current_threshold * self.threshold_rate
            else:
                threshold = self.current_threshold * (2-self.threshold_rate)
            threshold = max(0, threshold)
            threshold = min(1., threshold)
        return threshold

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

        return loss_ce + self.lamda * loss_ne

#-------------------------------
# In PICAnet, the only augementation way is to add gaussian noise to original data. 
class Divcnet(nn.Module):
    def __init__(self, feature_dim, num_classes,hidden_dims,lamda,noise_std,device,NMI_target=1., NMI_interval=20, threshold_rate=0.99):
        super(Divcnet, self).__init__()
        self.feature_dim=feature_dim
        self.num_classes=num_classes
        self.net =SimpleMLP(self.feature_dim,hidden_dims,self.num_classes)
        self.lamda=lamda
        self.noise_std=noise_std
        self.NMI_target=NMI_target
        self.NMI_interval=NMI_interval
        self.threshold_rate=threshold_rate
        self.PUILoss=PUILoss(lamda=self.lamda,device=device)
        self.device=device
        self=self.to(self.device)
        self.DivLoss = DivClustLoss(NMI_target=self.NMI_target,NMI_interval=self.NMI_interval,threshold_rate=self.threshold_rate)
        self.current_step = 0
        #print(self.parameters)

    def forward(self, x):  # shape=[n, c, w, h]
        noise = torch.randn_like(x) * self.noise_std
        noisy_data = x+ noise
        #print(x)
        z = self.net(x)
        z_noise=self.net(noisy_data)
        #print(z_noise)
        loss=self.PUILoss(z,z_noise)
        loss_div,_,_=self.DivLoss(z, self.current_step)
        loss=loss+loss_div
        if loss_div != 0.:
                loss = loss + loss_div
        self.current_step+=1
        return loss
    def predcit(self,x):
        z=self.net(x)
        return z


def train(model, dataset,epochs,stop_epochs, lr=1e-3,batch_size=256,device='cuda'):
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        return y_hat.detach().cpu()
            



class DIVC(BaseCluster):

    def __init__(self, n_clusters:int, 
                 hidden_dims:list = [512, 256, 128], stop_epochs: int = 50,
                 epochs: int = 300, lr: float = 1e-3, device: str = 'cuda',
                 lamda: float = 0.1, noise_std: float = 0.1,
                 batch_size: int = 256, NMI_target: float = 1.0, 
                 NMI_interval: int = 20, threshold_rate: float = 0.99,
                 final_epoch=True):
        super(DIVC, self).__init__()
        
        self.epochs = epochs
        self.lr = lr
        self.lamda=lamda
        self.hidden_dims=hidden_dims
        self.stop_epochs=stop_epochs
        self.noise_std=noise_std
        self.batch_size=batch_size
        self.device=device
        self.NMI_target=NMI_target
        self.NMI_interval=NMI_interval
        self.threshold_rate=threshold_rate
        self.n_clusters=n_clusters
        self.final_epoch = final_epoch

    def fit_predict(self, X):

        self.X = X
        self.input_dim = self.X.shape[1]
        self.Y = np.ones(self.X.shape[0])
        self.dataset =CustomDataset(self.X,self.Y)
        model= Divcnet(feature_dim=self.X.shape[1],hidden_dims=self.hidden_dims,num_classes=self.n_clusters,lamda=self.lamda,noise_std=self.noise_std,device=self.device,NMI_interval=self.NMI_interval,NMI_target=self.NMI_target,threshold_rate=self.threshold_rate)
        self.labels = train(model,self.dataset,epochs=self.epochs,stop_epochs=self.stop_epochs,lr=self.lr,batch_size=self.batch_size,device=self.device)

        if self.final_epoch:
            self.labels = self.labels[-1]
        self.time = time.time() - self.time
        return self.labels
    