import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.data_loader import swapCLR
from models.resnet_simclr import ResNetSimCLR, ResNetEncoder
from trainer_swapclr import SwapCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MaskCLR')
parser.add_argument('-comment', default='')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')

parser.add_argument('--dataset_name', default='vox1', type=str)
parser.add_argument('--training_mode', default='swapclr', type=str) 
parser.add_argument('--distr_mode', default='c', type=str,
                    choices=['a','b','c'], help='distribution of the positive pairs')
parser.add_argument('--time_aug', default=True, type=bool, help='Use time_augmentation or not')
parser.add_argument('--resume', default=False)
parser.add_argument('--state_dict', default='')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--threshold', default=0.5, type=float,
                    help='false negative threshold')
parser.add_argument('--negative_lambda', default=0.8, type=float,
                    help='parameter for gathering false negative')
parser.add_argument('--negative_weight', default=0.5, type=float)
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)
    img1_transforms, img2_transforms = dataset.get_cvimage_transform()

    # Here to load the dataset, test this function using test_data_loader
    train_dataset = swapCLR(args=args,
                             transforms_img1=img1_transforms,
                             transforms_img2=img2_transforms,)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=int(args.batch_size/2), shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, mode='image')
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)


    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        swapclr = SwapCLR(model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args)
        swapclr.train(train_loader)


if __name__ == "__main__":
    main()
