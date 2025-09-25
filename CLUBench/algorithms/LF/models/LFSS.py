import copy

import torch
from torch import nn
import sys
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import torch.nn.functional as F
#from models.Accuracy import clustering, best_match, test_torch_times
#from models.util import adjust_learning_rate
#from models.util import nt_xent, nt_xent_self, get_embedding_for_test
#from network import backbone_dict
from munkres import Munkres
#from utils.grad_scaler import NativeScalerWithGradNormCount
from torch.nn.functional import cosine_similarity
import numpy as np
import random
import torch_clustering
from torch import inf

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self,
                 optimizer=None,
                 amp=False,
                 clip_grad=None):
        self._scaler = torch.cuda.amp.GradScaler()
        self.clip_grad = clip_grad
        self.optimizer = optimizer
        self.amp = amp

    def __call__(self, loss, optimizer=None, clip_grad=None, parameters=None, update_grad=True, backward_kwargs={}):
        if optimizer is None:
            optimizer = self.optimizer
        if clip_grad is None:
            clip_grad = self.clip_grad
        if self.amp:
            self._scaler.scale(loss).backward(**backward_kwargs)
        else:
            loss.backward(**backward_kwargs)

        norm = None
        if update_grad:
            if self.amp:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if parameters is not None:
                    norm = get_grad_norm_(parameters)
            if self.amp:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        return norm



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
    #clustering_model=KMeans(max_iter=300,tol=1e-4,n_clusters=n_clusters,)

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

def collect_params(*models, exclude_bias_and_bn=True):
    param_list = []
    for model in models:
        for name, param in model.named_parameters():
            param_dict = {
                'name': name,
                'params': param,
            }
            if exclude_bias_and_bn and any(s in name for s in ['bn', 'bias']):
                param_dict.update({'weight_decay': 0., 'lars_exclude': True})
            param_list.append(param_dict)
    return param_list
def cosine_annealing_LR(opt, n_iter):

    epoch = n_iter / opt.num_batch + 1
    max_lr = opt.learning_rate
    min_lr = max_lr * opt.learning_eta_min
    # warmup
    if epoch < opt.warmup_epochs:
        # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr # 1
        lr = opt.learning_rate * epoch / opt.warmup_epochs 
    else:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - opt.warmup_epochs) * np.pi / opt.epochs))
    return lr

def step_LR(opt, n_iter):
    lr = opt.learning_rate
    epoch = n_iter / opt.num_batch
    if epoch < opt.warmup_epochs:
        # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr
        lr = opt.learning_rate * epoch / opt.warmup_epochs
    else:
        for milestone in opt.lr_decay_milestone:
            lr *= opt.lr_decay_gamma if epoch >= milestone else 1.
    return lr

def get_embedding_for_test(model,data_loader, mode = 'k'):
    model.eval()
    local_features = []
    local_labels = []
    for i, (idx,inputs, target) in enumerate(data_loader):
        with torch.no_grad():

            inputs = inputs.to(model.device)
            inputs=inputs.float()
            target = target.to(model.device)
            if mode == 'k':
                feature = model.encoder_k(inputs)
                feature = model.projector_k(feature)
            elif mode == 'q':
                feature = model.encoder_q(inputs)
                feature = model.projector_q(feature)
            elif mode == 'p':
                feature = model.encoder_q(inputs)
                feature = model.projector_q(feature)
                feature = model.predictor(feature)

            # feature = model.encoder_q(inputs)  # keys: NxC
            # feature = model.projector_q(feature)
            # feature = model.predictor(feature)
            local_features.append(feature)
            local_labels.append(target)
    features = torch.cat(local_features, dim=0)
    features = torch.nn.functional.normalize(features, dim=-1)
    labels = torch.cat(local_labels, dim=0)
    print(features.shape)
    print(labels.shape)
    return features, labels
def nt_xent(x, t=0.5, features2=None):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)
        # out_1 = x
        # out_2 = features2

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    # print("temperature is {}".format(t))
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss

    loss = (- torch.log(pos / (pos + Ng)))

    return loss.mean()
def nt_xent_self(x, t=0.5, features2=None):
    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    Ng = torch.sum(torch.exp(torch.mm(out_1, out_2.t().contiguous()) / t), dim=-1)
    loss = (- torch.log(pos / (Ng-pos)))

    return loss.mean()
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask
def adjust_learning_rate(opt, model, optimizer, n_iter):
    lr = cosine_annealing_LR(opt, n_iter)
    if opt.fix_predictor_lr:
        predictor_lr = opt.learning_rate
    else:
        predictor_lr = lr * opt.lambda_predictor_lr
    flag = False
    for param_group in optimizer.param_groups:
        if 'predictor' in param_group['name']:
            flag = True
            param_group['lr'] = predictor_lr
        else:
            param_group['lr'] = lr
    assert flag

    ema_momentum = opt.momentum_base
    if opt.momentum_increase:
        ema_momentum = opt.momentum_max - (opt.momentum_max - ema_momentum) * (
                np.cos(np.pi * n_iter / (opt.epochs * opt.num_batch)) + 1) / 2
    model.m = ema_momentum
    return lr

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

class LFSS(nn.Module):

    def __init__(self,momentum_base,dim_in,hidden_dims,num_cluster,temperature,fea_dim,sigma,lamb_da,hidden_size,amp,device):
        nn.Module.__init__(self)
        self.n_clusters=num_cluster
        encoder = SimpleMLP(fea_dim,hidden_dims,dim_in)
        print(encoder)
        self.dim_in=dim_in
        self.m = momentum_base
        self.num_cluster = num_cluster
        self.temperature = temperature
        self.device=device
        self.fea_dim = fea_dim
        self.sigma = sigma
        self.hidden_size=hidden_size
        self.lamb_da = lamb_da
        self.amp=amp

        # create the encoders
        self.encoder_q = encoder
        self.projector_q = nn.Sequential(
            nn.Linear(self.dim_in,self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.fea_dim)
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_k = copy.deepcopy(self.projector_q)

        self.predictor = nn.Sequential(nn.Linear(self.fea_dim, self.hidden_size),
                                       nn.BatchNorm1d(self.hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.hidden_size, self.fea_dim)
                                       )
        self.q_params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        self.k_params = list(self.encoder_k.parameters()) + list(self.projector_k.parameters())

        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder = nn.Sequential(self.encoder_k, self.projector_k)

        self.pre_encoder = copy.deepcopy(self.encoder)
        for param_o in list(self.pre_encoder.parameters()):
            param_o.requires_grad = False
        #if opt.syncbn:
        #    if opt.shuffling_bn:
        #        self.encoder_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        #        self.projector_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        #        self.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
        self.feature_extractor_copy = copy.deepcopy(self.encoder).cuda()

        self.scaler = NativeScalerWithGradNormCount(amp=self.amp)

        self.pseudo_labels = None
        self.num_cluster = num_cluster

    def get_old_embedding(self,im_q,im_k):
        q = self.pre_encoder(im_q)
        k = self.pre_encoder(im_k)

        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        return q, k
    def forward_noise_loss(self,q, k):

        noise_q = q + torch.randn_like(q) * self.sigma
        contrastive_loss = (2 - 2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()
        # contrastive_loss = (-2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()

        return contrastive_loss
    def forward_nt_loss(self, old, new):
        nt_xent_loss = nt_xent_self(new, 0.5, old)
        return nt_xent_loss
        # return self.forward_noise_loss(new,old)
    def forward_prt_loss(self, q, k, pseudo_labels):
        valid_index = torch.where(pseudo_labels!=-1)[0]
        n_samples = len(pseudo_labels)
        # print(n_samples)
        # print(pseudo_labels)
        weight = torch.zeros(self.num_cluster, n_samples).to(self.device)
        samples_index = torch.arange(n_samples).cuda()
        weight[pseudo_labels[valid_index].to(torch.long), samples_index[valid_index].to(torch.long)] = 1
        non_zero_mask = weight.any(dim=1)
        weight = weight[non_zero_mask]
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        q_centers = torch.mm(weight, q)
        q_centers = F.normalize(q_centers, dim=1)
        k_centers = torch.mm(weight, k)
        k_centers = F.normalize(k_centers, dim=1)
        loss = nt_xent(q_centers,0.5,k_centers)
        return loss
    def get_embedding(self, inputs, mode = 'k'):
        if mode == 'k':
            feature = self.encoder_k(inputs)
            feature = self.projector_k(feature)
        elif mode == 'q':
            feature = self.encoder_q(inputs)
            feature = self.projector_q(feature)
        elif mode == 'p':
            feature = self.encoder_q(inputs)
            feature = self.projector_q(feature)
            feature = self.predictor(feature)
        elif mode == 'q+p':
            feature = self.encoder_q(inputs)
            feature1 = self.projector_q(feature)
            feature2 = self.predictor(feature1)
            return feature1, feature2
        return feature
    def forward_nt(self, inputs1, inputs2, momentum_update=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        with torch.no_grad():
            old_1, old_2 = self.get_old_embedding(inputs1, inputs2)
        # 1
        loss_nt1 = self.forward_nt_loss(old_1, q_1)
        loss_nt2 = self.forward_nt_loss(old_2, q_2)
        loss_nt = (loss_nt2 + loss_nt1) * 0.5
        loss = loss_noise + loss_nt * self.lamb_da
        return loss
    def forward_prt(self, inputs1, inputs2, momentum_update = True, idx = None):

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        pseudo_labels = self.pseudo_labels[idx]
        loss_prt1 = self.forward_prt_loss(q_1, k_2, pseudo_labels)
        loss_prt2 = self.forward_prt_loss(q_2, k_1, pseudo_labels)
        loss_prt = (loss_prt2 + loss_prt1) * 0.5
        # print(loss_noise,loss_prt)
        loss = loss_noise + loss_prt * self.lamb_da
        return loss
    def forward_noise_only(self, inputs1, inputs2, momentum_update = True):

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        return loss_noise
    def forward_nt_prt(self,inputs1, inputs2, momentum_update = True, idx = None):

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        with torch.no_grad():
            old_1, old_2 = self.get_old_embedding(inputs1, inputs2)
        loss_nt1 = self.forward_nt_loss(old_1, q_1)
        loss_nt2 = self.forward_nt_loss(old_2, q_2)
        loss_nt = (loss_nt2 + loss_nt1) * 0.5
        pseudo_labels = self.pseudo_labels[idx]
        loss_prt1 = self.forward_prt_loss(q_1, k_2, pseudo_labels)
        loss_prt2 = self.forward_prt_loss(q_2, k_1, pseudo_labels)
        loss_prt = (loss_prt2 + loss_prt1) * 0.5
        # print(loss_noise,loss_prt,loss_nt)
        loss = loss_noise + loss_prt * self.lamb_da + loss_nt * self.lamb_da
        return loss
    def forward(self,inputs1, inputs2, epoch, eta, momentum_update=True, idx = None):
        if epoch <= eta:
            return self.forward_nt(inputs1, inputs2, momentum_update)
        else:
            return self.forward_nt_prt(inputs1, inputs2, momentum_update,idx)

    def get_pseudo_labels(self,data_loader,opt):
        features = torch.zeros([len(data_loader.dataset), opt.fea_dim]).cuda()
        old_features = torch.zeros([len(data_loader.dataset), opt.fea_dim]).cuda()
        for i, (inputs, target, idx) in enumerate(data_loader):
            with torch.no_grad():
                inputs_1, inputs_2, inputs_3 = inputs
                inputs_3 = inputs_3.cuda(non_blocking=True)

                feature = self.encoder_k(inputs_3)
                feature = self.projector_k(feature)
                features[idx] = feature
                old_feature = self.pre_encoder(inputs_3)
                old_features[idx] = old_feature
        y_pred, _ = clustering(features, opt.num_cluster)
        old_y_pred, _ = clustering(old_features, opt.num_cluster)
        y_pred = best_match(old_y_pred.cpu().numpy(), y_pred.cpu().numpy())
        y_pred = torch.from_numpy(y_pred).cuda()
        feature1 = torch.nn.functional.normalize(old_features, dim=-1)
        feature2 = torch.nn.functional.normalize(features, dim=-1)
        s = cosine_similarity(feature1, feature2, dim=1)
        sample_number = len(s)

        threshold, _ = torch.kthvalue(s, int(opt.delta * sample_number))
        change_idx = torch.where(s < threshold)[0]

        y_pred[change_idx] = -1
        print(len(y_pred), len(change_idx), len(change_idx) / len(y_pred))
        self.pseudo_labels = y_pred
        return y_pred

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


def train_LFSS(opt, model, optimizer, train_loader, epoch):
    n_iter = opt.num_batch * (epoch - 1) + 1
    if epoch > opt.eta and (((epoch - 1) % opt.prototype_freq == 0) or model.pseudo_labels == None):
        model.get_pseudo_labels(train_loader, opt)

    for i, (idx,inputs, target) in enumerate(train_loader):
        inputs_1, inputs_2, inputs_3 = inputs.float(),inputs.float(),inputs.float()
        inputs_1 = inputs_1.cuda(non_blocking=True)
        inputs_2 = inputs_2.cuda(non_blocking=True)
        inputs_3 = inputs_3.cuda(non_blocking=True)

        model.train()
        lr = adjust_learning_rate(opt, model, optimizer, n_iter)
        update_params = (n_iter % opt.acc_grd_step == 0)
        with torch.autocast(model.device, enabled=opt.amp):
            loss = model(inputs_1, inputs_2, epoch, opt.eta, update_params, idx)
        loss = loss / opt.acc_grd_step
        # model.pre_encoder = copy.deepcopy(model.encoder)
        model.scaler(loss, optimizer=optimizer, update_grad=update_params)
        with torch.no_grad():
            model._momentum_update_key_encoder()
        if i == 0:
            print('Epoch {} loss: {} lr: {}'.format(epoch, loss, lr))
        n_iter += 1
    model.pre_encoder = copy.deepcopy(model.encoder)

def test_LFSS(model,data_loader,class_num, mode='k'):
    features, labels = get_embedding_for_test(model,data_loader,mode)
    y_preds=test_torch_times(features, labels, 10, class_num)
    return y_preds




