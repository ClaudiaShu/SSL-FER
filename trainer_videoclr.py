import logging
import os
import sys
import random

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from loss.loss import CrossCLR_onlyIntraModality

torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.epoch = kwargs['epoch']
        self.args = kwargs['args']
        self.para = kwargs['para']
        if self.args.data_mode == 1:
            self.model = self.para['model_img'].to(self.args.device)
            self.optimizer = self.para['optimizer_img']
            self.scheduler = self.para['scheduler_img']
        elif self.args.data_mode == 2:
            self.model = self.para['model_aud'].to(self.args.device)
            self.optimizer = self.para['optimizer_aud']
            self.scheduler = self.para['scheduler_aud']
        elif self.args.data_mode == 3:
            self.model_img = self.para['model_img'].to(self.args.device)
            self.optimizer_img = self.para['optimizer_img']
            self.scheduler_img = self.para['scheduler_img']
            self.model_aud = self.para['model_aud'].to(self.args.device)
            self.optimizer_aud = self.para['optimizer_aud']
            self.scheduler_aud = self.para['scheduler_aud']

        # self.model = kwargs['model'].to(self.args.device)
        # self.optimizer = kwargs['optimizer']
        # self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        # Todo: Cross modality loss
        criterion = CrossCLR_onlyIntraModality(temperature=self.args.temp, negative_weight=self.args.weight_decay)
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'runs', current_time + '_' + socket.gethostname() + self.args.comment)
        dir2log = f'/mnt/d/Data/Yuxuan/logging/{self.args.training_mode}/{self.args.dataset_name}_frame{self.args.nb_frame}/{log_dir}'
        os.makedirs(dir2log, exist_ok=True)
        self.writer = SummaryWriter(log_dir=dir2log)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        best_acc = 0
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"training on {self.args.device}")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        logging.info(f"Training with {self.args.arch}")
        logging.info(f"Using dataset {self.args.dataset_name}")
        logging.info(f"Batch size: {self.args.batch_size}")
        logging.info(f"Initial learning rate: {self.args.lr}")
        logging.info(f"Time augmentation: {self.args.time_aug}")
        if self.args.time_aug:
            logging.info(f"With mode {self.args.distr_mode} temporal interval sampler")

        for epoch_counter in range(self.args.epochs):

            for Sample in tqdm(train_loader):
                # print()
                if self.args.data_mode == 1: # Only video
                    if self.args.training_mode == 'simclr':
                        Anchor = Sample['Anchor']
                        Positive = Sample['Positive']
                        data = torch.cat([Anchor, Positive], dim=0).to(self.args.device)
                    elif self.args.training_mode == 'videoclr':
                        Anchor = Sample['Anchor']
                        Positive = Sample['Positive']
                        NAnchor = Sample['NAnchor']
                        NPositive = Sample['NPositive']
                        data = torch.cat([Anchor, NAnchor, Positive, NPositive], dim=0).to(self.args.device)
                    # elif self.args.training_mode == 'onehardclr':
                    #     neg_id = random.randint(0, self.args.batch_size-2)
                    #     NAnchor = torch.unsqueeze(NAnchor[neg_id], dim=0)
                    #     NPositive = torch.unsqueeze(NPositive[neg_id], dim=0)
                    #     data = torch.cat([Anchor, NAnchor, Positive, NPositive], dim=0)
                    else:
                        raise ValueError
                    # Calculate loss
                    with autocast(enabled=self.args.fp16_precision):
                        features = self.model(x=data)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)
                elif self.args.data_mode == 2: # Only Audio
                    if self.args.training_mode == 'simclr':
                        AuAnchor = Sample['AuAnchor']
                        AuPositive = Sample['AuPositive']
                        data = torch.cat([AuAnchor, AuPositive], dim=0).to(self.args.device)
                    elif self.args.training_mode == 'videoclr':
                        AuAnchor = Sample['AuAnchor']
                        AuPositive = Sample['AuPositive']
                        AuNAnchor = Sample['AuNAnchor']
                        AuNPositive = Sample['AuNPositive']
                        data = torch.cat([AuAnchor, AuNAnchor, AuPositive, AuNPositive], dim=0).to(self.args.device)
                    else:
                        raise ValueError
                    # Calculate loss
                    with autocast(enabled=self.args.fp16_precision):
                        features = self.model(x=data)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)
                elif self.args.data_mode == 3: # image & audio
                    if self.args.training_mode == 'simclr':
                        Anchor = Sample['Anchor']
                        Positive = Sample['Positive']
                        AuAnchor = Sample['AuAnchor']
                        AuPositive = Sample['AuPositive']

                        data1 = Anchor.to(self.args.device)
                        data2 = Positive.to(self.args.device)
                        data3 = AuAnchor.to(self.args.device)
                        data4 = AuPositive.to(self.args.device)

                    elif self.args.training_mode == 'videoclr':
                        Anchor = Sample['Anchor']
                        Positive = Sample['Positive']
                        NAnchor = Sample['NAnchor']
                        NPositive = Sample['NPositive']
                        AuAnchor = Sample['AuAnchor']
                        AuPositive = Sample['AuPositive']
                        AuNAnchor = Sample['AuNAnchor']
                        AuNPositive = Sample['AuNPositive']

                        data1 = torch.cat([Anchor, NAnchor], dim=0).to(self.args.device)
                        data2 = torch.cat([Positive, NPositive], dim=0).to(self.args.device)
                        data3 = torch.cat([AuAnchor, AuNAnchor], dim=0).to(self.args.device)
                        data4 = torch.cat([AuPositive, AuNPositive], dim=0).to(self.args.device)

                    else:
                        raise ValueError
                    # Calculate loss
                    with autocast(enabled=self.args.fp16_precision):
                        # Todo: check whether the mel and pic could share weight
                        features1 = self.model_img(x=data1) # Anchor Image feature
                        features2 = self.model_img(x=data2) # Positive Image feature
                        features3 = self.model_aud(x=data3) # Anchor Audio feature
                        features4 = self.model_aud(x=data4) # Positive Audio feature
                        logits_12, labels_12 = self.info_nce_loss(torch.cat([features1, features2], dim=0))
                        logits_34, labels_34 = self.info_nce_loss(torch.cat([features3, features4], dim=0))
                        logits_13, labels_13 = self.info_nce_loss(torch.cat([features1, features3], dim=0))
                        # logits_12, labels_12 = self.info_nce_loss(torch.cat([features1, features2], dim=0))

                        loss_12 = self.criterion(logits_12, labels_12)
                        loss_34 = self.criterion(logits_34, labels_34)
                        loss_13 = self.criterion(logits_13, labels_13)
                        loss = loss_12 + loss_34 + loss_13
                else:
                    raise ValueError
                try:
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    scaler.update()
                except:
                    self.optimizer_img.zero_grad()
                    self.optimizer_aud.zero_grad()
                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer_img)
                    scaler.step(self.optimizer_aud)
                    scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    if self.args.data_mode == 3:
                        top1, top5 = accuracy(logits_12, labels_12, topk=(1, 5))
                        self.writer.add_scalars('modal_loss',
                                                {'loss_12':loss_12,
                                                 'loss_34':loss_34,
                                                 'loss_13':loss_13}, global_step=n_iter)
                    else:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalars('Accuracy',
                                            {'acc/top1':top1[0],
                                             'acc/top5':top5[0]}, global_step=n_iter)
                    self.writer.add_scalar('learning_rate/image', self.scheduler.get_lr()[0], global_step=n_iter)
                    # self.writer.add_scalar('learning_rate/audio', self.scheduler_aud.get_lr()[0], global_step=n_iter)

                n_iter += 1

            ###############################################################################################
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                try:
                    self.scheduler.step(self.epoch)
                except:
                    self.scheduler_img.step(self.epoch)
                    self.scheduler_aud.step(self.epoch)
            self.epoch += 1
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}")

            if epoch_counter%5 == 0:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                if self.args.data_mode == 3:
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict_img': self.model_img.state_dict(),
                        'optimizer_img': self.optimizer_img.state_dict(),
                        'scheduler_img': self.scheduler_img,
                        'state_dict': self.model_aud.state_dict(),
                        'optimizer': self.optimizer_aud.state_dict(),
                        'scheduler': self.scheduler_aud,
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

                else:
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler,
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

            if top1[0] >= best_acc:
                best_acc = top1[0]
                checkpoint_name = 'checkpoint_best.pth.tar'.format(epoch_counter)
                if self.args.data_mode == 3:
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict_img': self.model_img.state_dict(),
                        'optimizer_img': self.optimizer_img.state_dict(),
                        'scheduler_img': self.scheduler_img,
                        'state_dict': self.model_aud.state_dict(),
                        'optimizer': self.optimizer_aud.state_dict(),
                        'scheduler': self.scheduler_aud,
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                else:
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler,
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")