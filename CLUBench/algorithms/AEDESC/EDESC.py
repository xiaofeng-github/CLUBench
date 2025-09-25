from __future__ import print_function, division
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.nn import Linear
#import keras.backend as K
import warnings
import time
warnings.filterwarnings("ignore")


class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class D_constraint1(torch.nn.Module):

    def __init__(self,device):
        super(D_constraint1, self).__init__()
        self.device=device

    def forward(self, d):
        I = torch.eye(d.shape[1]).to(self.device)
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 	1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self,device):
        super(D_constraint2, self).__init__()
        self.device=device

    def forward(self, d, dim,n_clusters):
        S = torch.ones(d.shape[1],d.shape[1]).to(self.device)
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return 1e-3 * loss_d2_constraint


def seperate(Z, y_pred, n_clusters):
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    Z_seperate = seperate(Z, y_pred, n_clusters)
    Z_full = None
    U = np.zeros([n_clusters * d, n_clusters * d])
    print(U.shape)
    print("Initialize D")
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        print(Z_seperate[i].shape)
        u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
        print(u[:,0:d].shape)
        U[:,i*d:(i+1)*d] = u[:,0:d]
    D = U
    print("Shape of D: ", D.transpose().shape)
    print("Initialization of D Finished")
    return D

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # Encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)

        # Decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # Encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        # Decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z
   
class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,#input-dim
                 n_z,#embed-dim
                 n_clusters,
                 num_sample,
                 dataset,
                 eta=5,
                 d=5,
                 device='cuda',
                 pretrain_lr=1e-3,
                 pretrain_path=None):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters
        self.eta=eta
        self.d=d
        self.device=device
        self.dataset=dataset
        self.pretrain_lr=pretrain_lr
        self=self.to(self.device)

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z).to(self.device)

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

        
    def pretrain(self, path=''):
        if path == None:
            pretrain_ae(self.ae,self.pretrain_lr,self.dataset,self.device)
        # Load pre-trained weights
        #self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        
        x_bar, z = self.ae(x)
        d = self.d
        s = None
        eta = self.eta
      
        # Calculate subspace affinity
        for i in range(self.n_clusters):	
			
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)   
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

	# Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)     
        
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)
        
        # Constraints
        d_cons1 = D_constraint1(x.device)
        d_cons2 = D_constraint2(x.device)
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
  
        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2

        return total_loss

		
def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_ae(model,lr,dataset,device):

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        #torch.save(model.state_dict(), args.pretrain_path)
    #print("Model saved to {}.".format(args.pretrain_path))
    
def train_EDESC(x,y,n_input,n_clusters,num_sample,beta,d,n_z,lr,eta,pretrain_lr,epochs,hidden_dims,stop_epochs,device='cuda'):
    dataset=CustomDataset(torch.from_numpy(np.array(x, dtype=np.float32)),torch.from_numpy(y))

    model = EDESC(
        n_enc_1=hidden_dims[0],#500
        n_enc_2=hidden_dims[1],#500
        n_enc_3=hidden_dims[2],#1000
        n_dec_1=hidden_dims[0],
        n_dec_2=hidden_dims[1],
        n_dec_3=hidden_dims[2],
        n_input=n_input,
        n_z=n_z,
        n_clusters=n_clusters,
        dataset=dataset,
        num_sample = num_sample,
        pretrain_path=None,
        d=d,
        eta=eta,
        pretrain_lr=pretrain_lr
    )
    start = time.time()      

    # Load pre-trained model
    model.pretrain(None)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Cluster parameter initiate
    data = x
    y = y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10) 

    # Get clusters from Consine K-means 
    # ~ X = hidden.data.cpu().numpy()
    # ~ length = np.sqrt((X**2).sum(axis=1))[:,None]
    # ~ X = X / length
    # ~ y_pred = kmeans.fit_predict(X)
 
    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    print("Initial Cluster Centers: ", y_pred)
    
    # Initialize D
    D = Initialization_D(hidden, y_pred, n_clusters, d)
    D = torch.tensor(D).to(torch.float32)
    y_pred_last = y_pred
    model.D.data = D.to(device)
    
    model.train()
    preds=[]
    
    for epoch in range(epochs):
        x_bar, s, z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred      
        if (epoch+1)%stop_epochs==0 or epoch+1==epochs:
            preds.append(y_pred)
        

        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, z, pred=s, target=s_tilde, dim=d, n_clusters = n_clusters, beta = beta)
        print('Iter {}'.format(epoch),
                  ':LOSS {:.4f}'.format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Running time: ', end-start)
    return preds
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--eta', default=5, type=int)
    #parser.add_argument('--batch_size', default=512, type=int)    
    parser.add_argument('--dataset', type=str, default='reuters')
    parser.add_argument('--pretrain_path', type=str, default='data/reuters')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.dataset = 'reuters'
    if args.dataset == 'reuters':
        args.pretrain_path = 'data/reuters.pkl'
        args.n_clusters = 4
        args.n_input = 2000
        args.num_sample = 10000
    print(args)
    bestacc = 0 
    bestnmi = 0
    for i in range(10):
        acc, nmi = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi))
