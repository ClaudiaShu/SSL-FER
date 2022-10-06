import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.data_loader import videoCLR
from models.resnet_simclr import ResNetSimCLR
from trainer_videoclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch VideoCLR')
parser.add_argument('-comment', default='')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')

parser.add_argument('--dataset_name', default='vox1', type=str,
                    choices=['Aff2', 'vox1', 'vox2'], help='dataset use to train') #Aff2 vox1 vox2
parser.add_argument('--training_mode', default='videoclr', type=str,
                    choices=['simclr','videoclr'], help='method to train the network') # simclr videoclr
parser.add_argument('--distr_mode', default='c', type=str,
                    choices=['a','b','c'], help='distribution of the positive pairs')
parser.add_argument('--time_aug', default=False, type=bool, help='Use time_augmentation or not')
parser.add_argument('--nb_frame', default=1, type=int,
                    help='1 or n') # Number of video frames
parser.add_argument('--sec', default=3, type=int) # length of the audio sequence
parser.add_argument('--data_mode', default=1, type=int,
                    help='Choose between different mode, 1: only frame, 2: only audio, 3:frame & audio')
parser.add_argument('--resume', default=False)
parser.add_argument('--state_dict', default='')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
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
parser.add_argument('--temp', default=0.07)
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--negative_weight', default=0.5, type=float)
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


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
    img1_transforms, img2_transforms = dataset.get_image_transform()

    # Here to load the dataset, test this function using test_data_loader
    train_dataset = videoCLR(args=args,
                             transforms_img1=img1_transforms,
                             transforms_img2=img2_transforms)

    if args.training_mode == 'simclr':
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)
    elif args.training_mode == 'videoclr':
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=int(args.batch_size/2), shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)
    else:
        raise ValueError

    if args.data_mode == 1:
        model_img = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, mode='image')
        optimizer_img = torch.optim.Adam(model_img.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=len(train_loader), eta_min=0,
                                                                   last_epoch=-1)
        para = {
            'model_img': model_img,
            'optimizer_img': optimizer_img,
            'scheduler_img': scheduler_img
        }
    elif args.data_mode == 2:
        model_aud = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, mode='audio')
        optimizer_aud = torch.optim.Adam(model_aud.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler_aud = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_aud, T_max=len(train_loader), eta_min=0,
                                                                   last_epoch=-1)
        para = {
            'model_aud': model_aud,
            'optimizer_aud': optimizer_aud,
            'scheduler_aud': scheduler_aud
        }
    elif args.data_mode == 3:
        model_img = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, mode='image')
        optimizer_img = torch.optim.Adam(model_img.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        model_aud = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, mode='audio')
        optimizer_aud = torch.optim.Adam(model_aud.parameters(), args.lr, weight_decay=args.weight_decay)

        scheduler_aud = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_aud, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        para = {
            'model_img': model_img,
            'model_aud': model_aud,
            'optimizer_img': optimizer_img,
            'scheduler_img': scheduler_img,
            'optimizer_aud': optimizer_aud,
            'scheduler_aud': scheduler_aud
        }
    else:
        raise ValueError('data_mode invalid')
    epoch=0
    if args.resume:
        # epoch
        # model_state_dict = args.state_dict
        model_state_dict = "/mnt/d/Data/Yuxuan/logging/videoclr/vox1_frame1/runs/Jun12_18-24-09_ic_xiao/checkpoint_best.pth.tar"
        checkpoint = torch.load(model_state_dict, map_location='cuda:0')
        state_dict = checkpoint['state_dict_img']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        para['model_img'].load_state_dict(state_dict, strict=False)
        para['optimizer_img'].load_state_dict(checkpoint['optimizer_img'])
        # para['scheduler_img'].load_state_dict(checkpoint['scheduler_img'])

        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        para['model_aud'].load_state_dict(state_dict, strict=False)
        para['optimizer_aud'].load_state_dict(checkpoint['optimizer'])
        # para['scheduler_aud'].load_state_dict(checkpoint['scheduler'])
        epoch=checkpoint['epoch']

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(epoch=epoch, para=para, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
