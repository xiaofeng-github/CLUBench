from AEDESC.EDESC import train_EDESC
import numpy as np
from .base import BaseCluster
import time

class EDESC(BaseCluster):

    def __init__(self, n_clusters:int,stop_epochs=100,hidden_dims=[512,256,128], epochs: int = 300, lr: float = 1e-3, device: str = 'cuda',
                 beta:float=0.1,d:int =5, eta: int=5, pretrain_lr: float=1e-3, final_epoch=True):
        super(EDESC, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.stop_epochs=stop_epochs
        self.hidden_dims=hidden_dims
        self.device=device
        self.beta=beta
        self.d=d
        self.n_z=d*n_clusters
        self.eta=eta
        self.pretrain_lr=pretrain_lr
        self.n_clusters=n_clusters
        self.final_epoch = final_epoch

    def fit_predict(self, X):
        self.X = X
        self.input_dim = X.shape[1]
        self.Y = np.ones(X.shape[0])
        self.num_sample=self.X.shape[0]
        self.labels = train_EDESC(x=self.X,y=self.Y,epochs=self.epochs,n_input=self.input_dim,n_clusters=self.n_clusters,num_sample=self.num_sample,beta=self.beta,d=self.d,n_z=self.n_z,lr=self.lr,eta=self.eta,pretrain_lr=self.pretrain_lr,hidden_dims=self.hidden_dims,stop_epochs=self.stop_epochs)
        
        if self.final_epoch:
            self.labels = self.labels[-1]
        self.times = time.time() - self.times
        return self.labels
    