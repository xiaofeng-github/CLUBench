import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from .LF.models.LFSS import LFSS
from .LF.models.LFSS import train_LFSS
from .LF.models.LFSS import test_LFSS
from .LF.models.util import collect_params
from .base import BaseCluster
import time

class CustomDataset_idx(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx,self.data[idx], self.labels[idx]

class Opt:
    def __init__(self):
        # 训练基本参数
        self.save_freq = 100
        self.num_workers = 16
        self.resume_epoch = 0
        self.save_dir = ''
        self.checkpoint = ''
        self.resume = False
        self.dataset = 'cifar10'
        
        # 分布式和随机种子
        self.local_rank = 0
        self.seed = 0
        self.num_devices = 1
        
        # 优化器参数
        self.weight_decay = 0.0005
        self.momentum_base = 0.996
        self.momentum_max = 1
        self.momentum_increase = False
        self.amp = False
        self.exclude_bias_and_bn = False
        
        # 训练参数
        self.batch_size = 256
        self.epochs = 1000
        
        # 学习率参数
        self.learning_rate = 0.05
        self.learning_eta_min = 0.5
        self.lr_decay_gamma = 0.1
        self.lr_decay_milestone = [60, 80]
        self.step_lr = False
        self.fix_predictor_lr = False
        self.lambda_predictor_lr = 10
        self.momentum = 0.9
        self.scheduler = 'cosine'
        
        # 训练技巧参数
        self.acc_grd_step = 1
        self.warmup_epochs = 10
        self.dist = True
        self.hidden_size = 4096
        self.lars = False
        self.syncbn = True
        self.shuffling_bn = True
        self.temperature = 0.5
        self.fea_dim = 256
        self.reassign = 10
        self.sigma = 0.001
        self.delta = 0.1
        self.prototype_freq = 1
        self.lamb_da = 0.1
        self.eta = 200
        
        # 数据相关参数
        self.data_folder = '/datasets'
        self.test_resized_crop = False
        self.resized_crop_scale = 0.08
        self.use_gaussian_blur = False

class LFSSnet(object):
    def __init__(self, n_clusters: int,hidden_dims,stop_epochs,emb_dim=256, temp=0.5,
                 noise_std=0.01,lamda_da=0.1,hidden_size=4096,epochs: int = 100, 
                 lr: float = 1e-3, device: str = 'cuda',amp='store_true',batch_size=256):
        self.opt=Opt()

        self.n_clusters=n_clusters
        self.hidden_dims=hidden_dims
        self.stop_epochs=stop_epochs
        self.emb_dim=emb_dim
        self.temp=temp

        self.noise_std=noise_std
        self.lamda_da=lamda_da
        self.hidden_size=hidden_size
        self.epochs=epochs

        self.lr=lr
        self.device = device
        self.amp=amp
        self.opt.fea_dim=emb_dim
        self.batch_size=batch_size
        
     
    def fit_predict(self, X):

        self.model = LFSS(momentum_base=0.996,
                          dim_in=X.shape[1],
                          hidden_dims=self.hidden_dims,
                          fea_dim=self.emb_dim,num_cluster=self.n_clusters,
                          temperature=self.temp,sigma=self.noise_std,lamb_da=self.lamda_da,
                          hidden_size=self.hidden_size,amp=self.amp,device=self.device)
        self.dataset=CustomDataset_idx(X,np.ones(X.shape[0]))
        self.model=self.model.to(self.device)
        if self.opt.lars:
            from .LF.utils.optimizers import LARS
            optim = LARS
        else:
            optim = torch.optim.SGD
        optimizer = optim(params=collect_params(self.model, exclude_bias_and_bn=self.opt.exclude_bias_and_bn),
                      lr=self.lr, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        #train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
        train_loader = DataLoader(
        self.dataset,
        num_workers=0,
        batch_size=self.batch_size,
        shuffle=True,
        #sampler=train_sampler,
        pin_memory=False)
        preds=[]
        test_loader = DataLoader(
        self.dataset,
        num_workers=0,
        batch_size=self.batch_size,
        shuffle=False,
        pin_memory=False)
        self.opt.num_batch = len(train_loader)
        start_epoch = 1
        for epoch in range(start_epoch, self.epochs + 1):
            #train_sampler.set_epoch(epoch)
            train_LFSS(self.opt, self.model, optimizer, train_loader, epoch, device=self.device)
            if epoch %self.stop_epochs==0 or epoch==self.epochs:
                preds.append(test_LFSS(self.model,data_loader=test_loader,class_num=self.n_clusters))
        return preds


class CLUB_LFSS(BaseCluster):
    def __init__(self, n_clusters: int, hidden_dims=[512, 256, 128],stop_epochs=50,emb_dim=256, temp=0.5,
                 noise_std=0.001,lamda_da=0.1,hidden_size=4096,epochs: int = 100, 
                 lr: float = 1e-3, device: str = 'cuda',amp='store_true',batch_size=256, final_epoch=True):
       super(CLUB_LFSS, self).__init__()
       self.final_epoch = final_epoch 
       self.model=LFSSnet(
                        n_clusters=n_clusters,
                        hidden_dims=hidden_dims,
                        stop_epochs=stop_epochs,
                        emb_dim=emb_dim,
                        temp=temp,
                        noise_std=noise_std,
                        lamda_da=lamda_da,
                        hidden_size=hidden_size,
                        epochs=epochs,
                        lr=lr,
                        device=device,
                        amp=amp,
                        batch_size=batch_size)
       
    def fit_predict(self, X):
        y_preds=self.model.fit_predict(X)
        self.labels = y_preds
        if self.final_epoch:
            self.labels = y_preds[-1]
        self.time = time.time() - self.time
        return self.labels


# class LFSSnet(BaseCluster):
#     def __init__(self, n_clusters: int, hidden_dims=[512,256,128],stop_epochs=300,emb_dim=256, temp=0.5,noise_std=0.001,lamda_da=0.1,
#                  hidden_size=4096,epochs: int = 300, lr: float = 1e-3, device: str = 'cuda',amp='store_true',batch_size=256, final_epoch=True):
#         super(LFSSnet, self).__init__()
#         self.opt=Opt()
#         self.opt.num_cluster=n_clusters
#         self.stop_epochs=stop_epochs
#         self.device=device
#         self.batch_size=batch_size
#         self.emb_dim=emb_dim
#         self.n_clusters=n_clusters
#         self.hidden_dims=hidden_dims
#         self.temp=temp
#         self.noise_std=noise_std
#         self.lamda_da=lamda_da
#         self.hidden_size=hidden_size
#         self.n_clusters=n_clusters
#         self.epochs=epochs
#         self.amp=amp
#         self.lr=lr
#         self.final_epoch = final_epoch

#     def fit_predict(self, X):
#         self.opt.fea_dim=X.shape[1]
#         self.model =LFSS(momentum_base=0.996,dim_in=self.emb_dim,hidden_dims=self.hidden_dims,fea_dim=X.shape[1],
#                          num_cluster=self.n_clusters,temperature=self.temp,sigma=self.noise_std,
#                          lamb_da=self.lamda_da,hidden_size=self.hidden_size,amp=self.amp,device=self.device)
#         self.model=self.model.to(self.device)
#         self.dataset=CustomDataset_idx(X,np.ones(X.shape[0]))
#         if self.opt.lars:
#             from .LF.utils.optimizers import LARS
#             optim = LARS
#         else:
#             optim = torch.optim.SGD
#         optimizer = optim(params=collect_params(self.model, exclude_bias_and_bn=self.opt.exclude_bias_and_bn),
#                       lr=self.lr, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
#         #train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
#         train_loader = DataLoader(
#         self.dataset,
#         num_workers=0,
#         batch_size=self.batch_size,
#         shuffle=True,
#         #sampler=train_sampler,
#         pin_memory=False)
#         preds=[]
#         test_loader = DataLoader(
#         self.dataset,
#         num_workers=0,
#         batch_size=self.batch_size,
#         shuffle=False,
#         pin_memory=False)
#         self.opt.num_batch = len(train_loader)
#         start_epoch = 1
#         for epoch in range(start_epoch, self.epochs + 1):
#             #train_sampler.set_epoch(epoch)
#             train_LFSS(self.opt, self.model, optimizer, train_loader, epoch, device=self.device)
#             if epoch %self.stop_epochs==0 or epoch==self.epochs:
#                 preds.append(test_LFSS(self.model,data_loader=test_loader,class_num=self.n_clusters))
        
#         self.labels=preds
#         if self.final_epoch:
#             self.labels = preds[-1]
#         self.time = time.time() - self.time
#         return self.labels
