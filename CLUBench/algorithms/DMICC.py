import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import time
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from .base import BaseCluster
import sys
from torch.autograd import Function

class Loss_FMI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ff):

        norm_fx = ff / (ff ** 2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_fx.t(), norm_fx)
        k = coef_mat.size(0)
        lamb = 10
        EPS = sys.float_info.epsilon
        p_i = coef_mat.sum(dim=1).view(k, 1).expand(k, k)
        p_j = coef_mat.sum(dim=0).view(1, k).expand(k, k)
        p_i_j = torch.where(coef_mat < EPS, torch.tensor([EPS], device=coef_mat.device), coef_mat)
        p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        loss_fmi = (p_i_j * (torch.log(p_i_j) \
                          - (lamb + 1) * torch.log(p_j) \
                          - (lamb + 1) * torch.log(p_i))) / (k**2)

        loss_fmi = loss_fmi.sum()

        return loss_fmi

class Loss_ID(nn.Module):
    def __init__(self, tau2):
        super().__init__()
        self.tau2 = tau2

    def forward(self, x, y):

        L_id = F.cross_entropy(x, y)


        return L_id


def compute_joint(view1, view2):

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def crossview_contrastive_Loss(view1, view2, EPS=sys.float_info.epsilon):
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))
    lamb_1 = -10

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    #     Works with pytorch <= 1.2
    #     p_i_j[(p_i_j < EPS).data] = EPS
    #     p_j[(p_j < EPS).data] = EPS
    #     p_i[(p_i < EPS).data] = EPS

    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb_1 * torch.log(p_j) \
                      - lamb_1 * torch.log(p_i))

    loss = loss.sum()

    return loss

class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):

        tau = params[0].item()
        out = x.mm(memory.t())
        out.div_(tau)
        ctx.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, memory, y, params = ctx.saved_tensors
        tau = params[0]
        momentum = params[1]

        grad_output.div_(tau)

        grad_input = grad_output.mm(memory)
        grad_input.resize_as_(x)

        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(x.mul(1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None


class NonParametricClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, tau=1.0, momentum=0.5):
        super(NonParametricClassifier, self).__init__()
        self.register_buffer('params', torch.tensor([tau, momentum]))
        stdv = 1. / np.sqrt(input_dim / 3.)
        self.register_buffer(
            'memory',
            torch.rand(output_dim, input_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx],idx

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims,output_dim):
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
        # 展平输入（如果是图像等）
        
        return self.network(x)
    

#-------------------------------
# In PICAnet, the only augementation way is to add gaussian noise to original data. 
class DMnet(BaseCluster):
    def __init__(self, n_clusters,low_dim=64,hidden_dims=[512,256,128],
                 stop_epochs=50,noise_std=0.01,device='cuda',lamda1=0.00001,
                 lamda2=0.00001,batch_size=256,epochs=300,lr=1e-3, final_epoch=True):
        super(DMnet, self).__init__()
        self.lowdim=low_dim
        self.hidden_dims=hidden_dims
        self.stop_epochs=stop_epochs
        self.lamda1=lamda1
        self.lr=lr
        self.lamda2=lamda2
        self.noise_std=noise_std
        self.batch_size=batch_size
        self.epochs=epochs
        self.device=device
        self.norm = Normalize(2)
        self.n_clusters=n_clusters
        self.loss_id = Loss_ID(tau2=2.0)
        self.loss_fmi = Loss_FMI()
        self.norm= self.norm.to(device)
        self.final_epoch = final_epoch
        self.loss_id, self.loss_fmi = self.loss_id.to(self.device), self.loss_fmi.to(self.device)
    def forward(self,input):
        x_noisy = input + torch.randn_like(input) * self.noise_std
        # 获取特征
        features = self.net(input)
        features_noisy = self.net(x_noisy)
        # 归一化特征
        norm_features = self.norm(features)
        norm_features_noisy = self.norm(features_noisy)
        
        return norm_features, norm_features_noisy
    # lr = 0.03
        #print(self.parameters)
    def fit_predict(self,X):
        self.feature_dim=X.shape[1]
        self.X=torch.from_numpy(X)
        self.y=torch.ones(len(X))
        self.dataset=CustomDataset(X,self.y)
        self.npc = NonParametricClassifier(input_dim=self.lowdim,
                                  output_dim=len(self.X),
                                  tau=1.0,
                                  momentum=0.5).to(self.device)
        self.net =SimpleMLP(self.feature_dim,self.hidden_dims,self.lowdim).to(self.device)
        optimizer = torch.optim.SGD(self.net.parameters(),
                                lr=0.05,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=False,
                                dampening=0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        [500, 950, 1350, 2050, 2350, 2750, 3350, 3750, 4250, 4550],
                                                        gamma=0.5)
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                              drop_last=False)
        preds=[]
        for epoch in range(0, self.epochs + 1):
            self.net.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
            for pos_1,target, index in train_bar:
                optimizer.zero_grad()
                inputs_1 = pos_1.to(self.device, dtype=torch.float32, non_blocking=True)
                indexes = index.to(self.device, non_blocking=True)
                features_1,features_2= self.forward(inputs_1)
                #print("AAA")
                outputs = self.npc(features_1, indexes)

                loss_imi = crossview_contrastive_Loss(features_1, features_2)

                loss_id = self.loss_id(outputs, indexes)
                #print(loss_id)
                loss_fmi = self.loss_fmi(features_1)
                #print(loss_fmi)
                #loss_fmi = loss_fmi(features_2)

                tot_loss = loss_id + self.lamda1 * loss_fmi + self.lamda2* loss_imi
                #print("EPOCH: "+str(epoch)+" LOSS:" +str(tot_loss.item()))
                tot_loss.backward()
                optimizer.step()
            if (epoch+1)%self.stop_epochs==0 or epoch+1==self.epochs:
                preds.append(self.testing())
            lr_scheduler.step()
        self.labels = preds
        if self.final_epoch:
            self.labels = preds[-1]
        self.times = time.time() - self.times
        return self.labels

    
    def testing(self):
        trainFeatures = self.npc.memory
        z = trainFeatures.cpu().numpy()
        n_clusters = self.n_clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z)
        return y_pred