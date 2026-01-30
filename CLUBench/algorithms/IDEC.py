import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import Tuple, Callable, Optional, Union
from tqdm import tqdm
from typing import Optional
from scipy.optimize import linear_sum_assignment
from torch.nn import Parameter
import torch.nn.functional as F
from .utils import CustomDataset_idx, SimpleAE, TwoLayerDAE
from copy import deepcopy
import time
from .base import BaseCluster
from torch.optim import Adam


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class ORIIDEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        alpha: float = 1.0,
        gamma:float=1.0,
        update_interval:int=10

    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param gamma: parameter representing the coefficient of the KL-loss, default 1.0.
        """
        super(ORIIDEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.gamma=gamma
        self.decoder=decoder
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )
        self.update_interval=update_interval

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.encoder(batch))
    def recon(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.decoder(self.encoder(batch))

def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def train(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    epochs: int,
    stop_epochs:int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    stopping_delta: Optional[float] = None,
    collate_fn=default_collate,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: int = 10,
    evaluate_batch_size: int = 1024,
    update_callback: Optional[Callable[[float, float], None]] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    device: str = 'cuda'
) -> None:
    """
    Train the DEC model given a dataset, a model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to train
    :param epochs: number of training epochs
    :param batch_size: size of the batch to train with
    :param optimizer: instance of optimizer to use
    :param stopping_delta: label delta as a proportion to use for stopping, None to disable, default None
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether to use CUDA, defaults to True
    :param sampler: optional sampler to use in the DataLoader, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, None disables, default 10
    :param evaluate_batch_size: batch size for evaluation stage, default 1024
    :param update_callback: optional function of accuracy and loss to update, default None
    :param epoch_callback: optional function of epoch and model, default None
    :return: None
    """
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=True,
    )
    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
        },
        disable=silent,
    )
    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    actual = []
    # form initial cluster centres
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 3:
            _,batch, value = batch  # if we have a prediction label, separate it to actual
            actual.append(value)
            # batch = batch.cuda(non_blocking=True)
        batch = batch.to(device)
        features.append(model.encoder(batch).detach().cpu())
    actual = torch.cat(actual).long()
    print(torch.cat(features).numpy().shape)
    predicted = kmeans.fit_predict(torch.cat(features).numpy())
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
        # cluster_centers = cluster_centers.cuda(non_blocking=True)
    cluster_centers = cluster_centers.to(device)
    print(cluster_centers.shape)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    loss_function = nn.KLDivLoss(size_average=False)
    delta_label = None
    preds=[]
    for epoch in range(epochs):
        features = []
        data_iterator = tqdm(
            train_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "acc": "%.4f" % (accuracy or 0.0),
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % (delta_label or 0.0),
            },
            disable=silent,
        )
        st_iterator = tqdm(
            static_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": epoch,
                "acc": "%.4f" % (accuracy or 0.0),
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % (delta_label or 0.0),
            },
            disable=silent,
        )
        model.train()
        if epoch % model.update_interval==0:
            print("Reset P")
            targets=None
            for index, batch in enumerate(st_iterator):
              if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 3:
                _,batch, value = batch  # if we have a prediction label, separate it to actual
              batch = batch.to(device)
              if targets==None:
                  output = model(batch)
                  targets=target_distribution(output).detach()
                  #print(targets.shape)
              else:
                  output = model(batch)
                  targets=torch.concat([targets,target_distribution(output).detach()],dim=0)
                  #print(targets.shape)

        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                batch
            ) == 3:
                idx,batch, _ = batch  # if we have a prediction label, strip it away
            #if cuda:
                # batch = batch.cuda(non_blocking=True)
            batch = batch.to(device)
            output = model(batch)
            recon_output=model.recon(batch)
            MSE=F.mse_loss(recon_output, batch, reduction='sum')

            target = targets[idx]
            loss = loss_function(output.log(), target) / output.shape[0]+MSE*model.gamma
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % float(loss.item()),
                dlb="%.4f" % (delta_label or 0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            features.append(model.encoder(batch).detach().cpu())
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % loss_value,
                    dlb="%.4f" % (delta_label or 0.0),
                )
                if update_callback is not None:
                    update_callback(accuracy, loss_value, delta_label)
        if (epoch+1)%stop_epochs==0 or epoch+1==epochs:
            predicted, actual = predict(
            dataset,
            model,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            return_actual=True,
            cuda=cuda,
            device=device
        )
            preds.append(predicted)
        data_iterator.set_postfix(
            epo=epoch,
            acc="%.4f" % ( 0.0),
            lss="%.8f" % 0.0,
            dlb="%.4f" % (0.0),
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model)
    return preds


def predict(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    cuda: bool = True,
    silent: bool = False,
    return_actual: bool = False,
    device: str = 'cuda'
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Predict clusters for a dataset given a DEC model instance and various configuration parameters.

    :param dataset: instance of Dataset to use for training
    :param model: instance of DEC model to predict
    :param batch_size: size of the batch to predict with, default 1024
    :param collate_fn: function to merge a list of samples into mini-batch
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param return_actual: return actual values, if present in the Dataset
    :return: tuple of prediction and actual if return_actual is True otherwise prediction
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent,)
    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 3:
            _,batch, value = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError(
                "Dataset has no actual value to unpack, but return_actual is set."
            )
        #if cuda:
            # batch = batch.cuda(non_blocking=True)
        batch = batch.to(device)
        features.append(
            model(batch).detach().cpu()
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features).max(1)[1]


class IDEC(BaseCluster):

    def __init__(self, n_clusters: int, 
                 hidden_dims: list = [512, 256, 128], stop_epochs: int = 50,
                 emb_dim: int = 64, epochs: int = 300, batch_size: int = 256, 
                 lr: float = 1e-3, device: str = 'cuda',gamma: float = 0.1,
                 update_interval: int=10, final_epoch=True):
        super(IDEC, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.update_interval = update_interval
        self.hidden_dims = deepcopy(hidden_dims)
        self.emb_dim = emb_dim
        self.hidden_dims.append(self.emb_dim)
        self.device = device
        self.stop_epochs=stop_epochs
        self.n_clusters = n_clusters
        self.final_epoch = final_epoch
        
 
    def _initial_ae(self, model, input_dim):

        dims = deepcopy(model.hidden_dims)
        dims.insert(0, input_dim)

        h = deepcopy(self.X)
        for layer_i, i in enumerate(range(len(dims) - 1)):
            # print(f'pretrain DAE for SAE layer [{i + 1}]')
            if layer_i == 0:
                h, dae_encoder, dae_decoder = self._train_dae(X=h, input_dim=dims[i], hidden_dim=dims[i + 1], g2='leaky_relu')
            elif layer_i == len(dims) - 2:
                h, dae_encoder, dae_decoder = self._train_dae(X=h, input_dim=dims[i], hidden_dim=dims[i + 1], g1='leaky_relu')
            else:
                h, dae_encoder, dae_decoder = self._train_dae(X=h, input_dim=dims[i], hidden_dim=dims[i + 1])
            encoder_layer_name = f'encoder.encoder_fc{layer_i + 1}'
            decoder_layer_name = f'decoder.decoder_fc{layer_i + 1}'
            # print(model.state_dict().keys())
            # print(dae_encoder.state_dict().keys())
            # print(dae_decoder.state_dict().keys())
            model.state_dict()[encoder_layer_name + '.weight'].copy_(dae_encoder.state_dict()['0.weight'])
            model.state_dict()[encoder_layer_name + '.bias'].copy_(dae_encoder.state_dict()['0.bias'])
            model.state_dict()[decoder_layer_name + '.weight'].copy_(dae_decoder.state_dict()['0.weight'])
            model.state_dict()[decoder_layer_name + '.bias'].copy_(dae_decoder.state_dict()['0.bias'])
        return model

    def _train_dae(self, X, input_dim, hidden_dim, g1='relu', g2='relu', dropout_rate=0.2):

        model = TwoLayerDAE(input_dim, hidden_dim, g1=g1, g2=g2, dropout_rate=dropout_rate)
        epochs = 200
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        model = model.to(self.device)
        train_data = DataLoader(dataset=X, batch_size=500, shuffle=True)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for batch in train_data:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                _, x_hat = model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            # print(f"Pretrain DAE Epoch {epoch}/{epochs}, Loss: {epoch_loss}")
        h, _ = model(X.to(self.device))
        return h.cpu().data, model.encoder, model.decoder

        
    def get_encoder(self, hidden_dims):

        train_data = DataLoader(dataset=self.X, batch_size=500, shuffle=True)
        model = SimpleAE(input_dim=self.input_dim, hidden_dims=hidden_dims)

        model = self._initial_ae(model, self.input_dim)
        epochs = 200
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        model = model.to(self.device)
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for batch in train_data:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                x_hat = model(batch)
                loss = criterion(x_hat, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(f"Pretrain SAE Epoch {epoch}/{epochs}, Loss: {epoch_loss}")

        return model.encoder, model.decoder,model.hidden_dims[-1]
    

    def fit_predict(self, X):

        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.input_dim = self.X.shape[1]
        self.Y = torch.ones(self.X.shape[0])
        self.data = CustomDataset_idx(self.X, self.Y)
        self.encoder, self.decoder,self.hidden_dim = self.get_encoder(hidden_dims=self.hidden_dims)
        self.model = ORIIDEC(cluster_number=self.n_clusters, hidden_dimension=self.emb_dim, encoder=self.encoder,decoder=self.decoder,gamma=self.gamma,update_interval=self.update_interval)
        
        
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.labels = train(dataset=self.data, model=self.model.to(self.device), optimizer=optimizer, stop_epochs=self.stop_epochs,
                                   epochs=self.epochs, batch_size=self.batch_size, device=self.device)
                
        if self.final_epoch:
            self.labels = self.labels[-1]
        self.time = time.time() - self.time
        return self.labels
    