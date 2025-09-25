
import argparse
import os

import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import ImageFile



from models.LFSS import test_LFSS, LFSS
from models.util import get_dataset, collect_params

ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser('Default arguments for training of different methods')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
parser.add_argument('--resume_epoch', type=int, default=0, help='number of training epochs')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--resume', default=True, type=bool, help='if resume training')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')


parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_devices', type=int, default=1, help='number of devices to use')


parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--momentum_base', type=float, default=0.996, help='momentum')
parser.add_argument('--momentum_max', type=float, default=1, help='momentum')
parser.add_argument('--momentum_increase', help='momentum_increase', action='store_true')
parser.add_argument('--amp', action='store_true', help='amp')
parser.add_argument('--exclude_bias_and_bn', help='exclude_bias_and_bn', action='store_true')


parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')


parser.add_argument('--learning_rate', type=float, default=0.05, help='base learning rate')
parser.add_argument('--learning_eta_min', type=float, default=0., help='minimum learning rate')
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
parser.add_argument('--lr_decay_milestone', nargs='+', type=int, default=[60, 80])
parser.add_argument('--step_lr', action='store_true', help='step_lr')
parser.add_argument('--fix_predictor_lr', help='fix the lr of predictor', action='store_true')
parser.add_argument('--lambda_predictor_lr', help='fix the lr of predictor', type=float, default=10)
parser.add_argument('--momentum', type=float, default=0.9, help='lr momemtum')
parser.add_argument('--scheduler', type=str, default='cosine', help='scheduler')


parser.add_argument('--acc_grd_step', type=int, default=1)#1
parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs') 
parser.add_argument('--dist', action='store_false', help='use for clustering')
parser.add_argument('--hidden_size', help='hidden_size', type=int, default=4096)
parser.add_argument('--lars', help='lars', action='store_true')
parser.add_argument('--syncbn', help='syncbn', action='store_false')
parser.add_argument('--shuffling_bn', help='shuffling_bn', action='store_false')
parser.add_argument('--temperature', help='temperature', type=float, default=0.5)
parser.add_argument('--fea_dim', type=int, default=256, help='projection fea_dim') 
parser.add_argument('--reassign', type=int, default=10, help='reassign kmeans')
parser.add_argument('--sigma', type=float, default=0.001, help='noise intensity')
parser.add_argument('--delta', type=float, default=0.1, help='unstable ratio') 
parser.add_argument('--prototype_freq', type=int, default=1, help='prototype generating frequency')
parser.add_argument('--lamb_da', type=int, default=0.1, help='balance parameter')
parser.add_argument('--eta', type=int, default=0, help='begin prototype')

parser.add_argument('--data_folder', type=str, default='/datasets', help='path to custom dataset')
parser.add_argument('--test_resized_crop', action='store_true', help='imagenet test transform')

parser.add_argument('--resized_crop_scale', type=float, default=0.08, help='randomresizedcrop scale')
parser.add_argument('--use_gaussian_blur', action='store_true', help='use_gaussian_blur')





def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    global opt
    opt = parser.parse_args()
    if opt.dist:
        dist.init_process_group(backend='nccl', init_method='env://' )#f'tcp://localhost:10001?rank={rank}&world_size={world_size}')
        torch.cuda.set_device(dist.get_rank())
    if opt.num_devices > 0:
        assert opt.num_devices == torch.cuda.device_count()  # total batch size
    if os.path.exists(opt.save_dir) is not True:
        os.system("mkdir -p {}".format(opt.save_dir))
    seed = opt.seed

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    rank = torch.distributed.get_rank()
    # logName = "log.txt"
    # log = logger(path=opt.save_dir, local_rank=rank, log_name=logName)
    # log.info(str(opt))
    if opt.dataset =='cifar10':
        opt.img_size = 32
        opt.num_cluster = 10
        opt.encoder_name = "bigresnet18"
    elif opt.dataset == 'cifar20':
        opt.img_size = 32
        opt.num_cluster = 20
        opt.encoder_name = 'bigresnet18'
    elif opt.dataset == 'stl10':
        opt.img_size = 96
        opt.num_cluster = 10
        opt.encoder_name = 'resnet18'

    elif opt.dataset == 'imagenet10':
        opt.img_size = 96
        opt.num_cluster = 10
        opt.encoder_name = 'resnet18'
        opt.test_resized_crop = True

    elif opt.dataset == 'imagenetdogs':
        opt.img_size = 96
        opt.num_cluster = 15
        opt.encoder_name = 'resnet18'
        opt.test_resized_crop = True
    elif opt.dataset == 'tiny-imagenet':
        opt.img_size = 64 #96
        opt.num_cluster = 200
        opt.encoder_name = 'resnet18'
        opt.test_resized_crop = True
    elif opt.dataset == 'imagenet':
        opt.img_size = 224 #96
        opt.num_cluster = 1000
        opt.encoder_name = 'resnet50'
    else:
        print("unknown dataset")
    model = LFSS(opt)
    model.cuda()
    # model.convert_to_ddp(model)
    if opt.lars:
        from utils.optimizers import LARS

        optim = LARS
    else:
        optim = torch.optim.SGD
    optimizer = optim(params=collect_params(model, exclude_bias_and_bn=opt.exclude_bias_and_bn),
                      lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    train_datasets = get_dataset(opt,'test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=False)
    opt.num_batch = len(train_loader)
    start_epoch = 0
    if opt.resume or opt.checkpoint != '':
        if opt.checkpoint == '':
            checkpoint = torch.load(os.path.join(opt.save_dir, 'model.pt'), map_location="cuda")
        else:
            checkpoint = torch.load(os.path.join(opt.save_dir, opt.checkpoint), map_location="cuda")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            # log.info("resume the checkpoint {} from epoch {}".format(opt.checkpoint, checkpoint['epoch']))
        else:
            # log.info("cannot resume since lack of files")
            assert False

    test_LFSS(model,train_loader,opt.num_cluster)

