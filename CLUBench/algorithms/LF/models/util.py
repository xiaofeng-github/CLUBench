import os
from os import path as osp

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
import sys
sys.path.append((osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
# from data.STL10 import CustomSTL10
# from data.cifar10 import CustomCIFAR10
# from data.cifar100 import CustomCIFAR100
# from data.imagenet import CustomImagenet
#from utils import TwoCropTransform, MemTransform


def normalize(dataset_name):
    normalize_params = {
        'cifar10': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
        'cifar20': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'stl10': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }
    if dataset_name not in normalize_params.keys():
        mean, std = normalize_params['imagenet']
        print(f'Dataset {dataset_name} does not exist in normalize_params,'
              f' use default normalizations: mean {str(mean)}, std {str(std)}.')
    else:
        mean, std = normalize_params[dataset_name]

    normalize = transforms.Normalize(mean=mean, std=std, inplace=True)
    return normalize
def transform(opt,type):
    normalization = normalize(opt.dataset)
    if type == 'train':
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if opt.use_gaussian_blur:
            train_transform.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5)
            )

        train_transform += [transforms.ToTensor(), normalization]
        # train_transform += [transforms.ToTensor()]

        train_transform = transforms.Compose(train_transform)

        train_transform = TwoCropTransform(train_transform)
        return train_transform
    elif type == 'test':
        def resize(image):
            size = (opt.img_size, opt.img_size)
            if image.size == size:
                return image
            return image.resize(size)

        test_transform = []
        if opt.test_resized_crop:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        test_transform += [
            resize,
            transforms.ToTensor(),
            normalization
        ]

        test_transform = transforms.Compose(test_transform)
        return test_transform
    elif type == 'mem':
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if opt.use_gaussian_blur:
            train_transform.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5)
            )

        train_transform += [transforms.ToTensor(), normalization]
        # train_transform += [transforms.ToTensor()]

        train_transform = transforms.Compose(train_transform)
        def resize(image):
            size = (opt.img_size, opt.img_size)
            if image.size == size:
                return image
            return image.resize(size)

        test_transform = []
        if opt.test_resized_crop:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        test_transform += [
            resize,
            transforms.ToTensor(),
            normalization
        ]

        test_transform = transforms.Compose(test_transform)
        return MemTransform(train_transform,test_transform)

def get_dataset(opt,type='train'):
    dataset_name=opt.dataset
    root = opt.data_folder
    tfs = transform(opt,type)
    if dataset_name == 'cifar10':
        dataset = CustomCIFAR10(root=root, train=True, transform=tfs, download=True)
    elif dataset_name == 'cifar20':
        dataset = CustomCIFAR100(root=root, train=True, transform=tfs, download=True)
    elif dataset_name == 'stl10':
        if type == 'test':
            dataset = CustomSTL10(root=root,  split='train+test', transform=tfs, download=True)
        else:
            dataset = CustomSTL10(root=root,  split='train+test+unlabeled', transform=tfs, download=True)

    elif dataset_name == 'imagenet10' or dataset_name == 'imagenetdogs':
        dataset = CustomImagenet(root=os.path.join(root,dataset_name), transform=tfs)
    elif dataset_name == 'imagenet':
        dataset = CustomImagenet(root='/datasets/imagenet/train', transform=tfs)
    elif dataset_name == 'tiny-imagenet':
        dataset = CustomImagenet(root='/datasets/tiny-imagenet-200/train', transform=tfs)

    return dataset
class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")

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

            inputs = inputs.to('cuda')
            inputs=inputs.float()
            target = target.to('cuda')
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
