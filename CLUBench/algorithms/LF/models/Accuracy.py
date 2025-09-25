import random

import numpy as np
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score

import torch_clustering

def cluster_accuracy(y_true, y_pre,verbose=True):
    y_best = best_match(y_true, y_pre)
    # for c in np.unique(y_true):
    #     print([c,np.sum(y_true==c),np.sum(y_best==c)])
    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = normalized_mutual_info_score(y_true, y_pre)
    ari = adjusted_rand_score(y_true, y_pre)
    F=best_cal(y_best,y_true)
    if verbose:
        print(F.astype(int))
    return acc, nmi, ari
def best_cal(y_best,y_true):
    F=np.zeros([np.unique(y_true).shape[0],np.unique(y_true).shape[0]])
    for i in range(y_best.shape[0]):
        # print([y_true[i],y_best[i]])
        F[int(y_true[i])][int(y_best[i])]+=1
    return F

def best_match(y_true, y_pre):
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]
    return y_best

def clustering(features, n_clusters,random_state=0):

    kwargs = {
        'metric': 'cosine' ,#if self.l2_normalize else 'euclidean',
        'distributed': True, #True
        'random_state': random_state,
        'n_clusters': n_clusters,
        'verbose': False
    }
    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

    psedo_labels = clustering_model.fit_predict(features)
    cluster_centers = clustering_model.cluster_centers_
    return psedo_labels, cluster_centers
def test_torch_times(embedding,target,times,class_num):
    random.seed(42) # 42
    times=1
    random_numbers = [random.randint(0, 10000) for _ in range(times)]# 10000
    ACC=[]
    NMI=[]
    ARI=[]
    # METRIC = []
    target = np.asarray(target.cpu())
    cnt=0
    for i in random_numbers:
        print(cnt)
        cnt+=1
        y_pred, _ = clustering(embedding, class_num,random_state=i)
        y_pred = np.asarray(y_pred.cpu())
        #acc, nmi, ari = cluster_accuracy(target, y_pred,verbose=False)
        #ACC.append(acc)
        #NMI.append(nmi)
        #ARI.append(ari)
    return y_pred
