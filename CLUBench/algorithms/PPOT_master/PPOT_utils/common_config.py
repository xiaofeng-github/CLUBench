"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
# import os
import math
import numpy as np
import torch
# import torchvision.transforms as transforms
from .collate import collate_custom
from torch.utils.data import Dataset
import torch.nn as nn

class Tabular_Dataset(Dataset):

    def __init__(self, X):

        super(Tabular_Dataset, self).__init__()
        self.data=torch.from_numpy(X).float()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], 1
        img_size = img.shape
        # class_name = self.classes[target]

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        return True


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

def get_model(p):
    # Get backbone
    if p['backbone'] == 'dino_vitb16':
        #from models.dino import get_dino_vitb16
        hidden_dims=p['hidden_dims']
        hidden_dims.append(p['emb_dim'])
        backbone = SimpleMLP(p['input_dim'],hidden_dims)
        try:
            p['model_kwargs']['features_dim'] = p['emb_dim']
        except:
            pass
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    from models.models import ClusteringModel
    model = ClusteringModel(backbone, p['emb_dim'],p['num_classes'], p['num_heads'], head_type=p['head_type'] if 'head_type' in p else 'linear')
    return model


def get_train_dataset(p, transform, to_augmented_dataset=False,
                        split="train"):

    if p['train_db_name'] == 'cifar_im':
        from data.cifar import get_imbalance_cifar
        dataset = get_imbalance_cifar(num_classes=p['num_classes'][0],imbalance_ratio=p['imbalance_ratio'],transform=transform,split=split)

    elif p['train_db_name'] == 'iNature_im':
        from data.inature import get_inaturelist18_datasets
        dataset = get_inaturelist18_datasets(train_transform=transform,split=split,num_classes=p['num_classes'][0])

    elif p['train_db_name'] == 'imagenet-r_im':
        assert p['num_classes'][0] == 200
        from data.imagenet import get_ImageNet_datasets
        dataset = get_ImageNet_datasets(num_classes=p['num_classes'][0],imbalance_factor=p['imbalance_ratio'],transform_train=transform,split=split,version=p['train_db_name'])


    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)
    
    return dataset

def get_val_dataset(p, transform=None):
    # Base dataset
    if p['val_db_name'] == 'cifar_im':
        from data.cifar import get_imbalance_cifar
        dataset = get_imbalance_cifar(num_classes=p['num_classes'][0],imbalance_ratio=p['imbalance_ratio'],transform=transform,split="val")

    elif p['val_db_name'] == 'iNature_im':
        from data.inature import get_inaturelist18_datasets
        dataset = get_inaturelist18_datasets(test_transform=transform,split="val",num_classes=p['num_classes'][0])

    elif p['val_db_name'] == 'imagenet-r_im':
        from data.imagenet import get_ImageNet_datasets
        dataset = get_ImageNet_datasets(num_classes=p['num_classes'][0],imbalance_factor=p['imbalance_ratio'],transform_val=transform,split="val",version=p['val_db_name'])
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batchSize'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['eval_batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['backbone'] == 'dino_vitb16':
        from torchvision.transforms import InterpolationMode
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = InterpolationMode.BICUBIC
        crop_pct = 0.875
        image_size=224
        return transforms.Compose([
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
    else:
        raise NotImplementedError("unknown backbone")


def get_val_transformations(p):
    if p['backbone'] == 'dino_vitb16':
        from torchvision.transforms import InterpolationMode
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation=InterpolationMode.BICUBIC
        crop_pct = 0.875
        image_size=224
        return transforms.Compose([
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
    else:
        raise NotImplementedError("unknown backbone")


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
            print(name)
            if 'cluster_head' in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=p['lr'], weight_decay=p['weight_decay'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=p['lr'], weight_decay=p['weight_decay'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
