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

torch.manual_seed(0)

class MaskCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.para = kwargs['para']
        self.epoch = kwargs['epoch']
        self.encoder = kwargs['encoder'].to(self.args.device)
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'runs', current_time + '_' + socket.gethostname() + self.args.comment)
        dir2log = f'/mnt/d/Data/Yuxuan/logging/{self.args.training_mode}/{self.args.dataset_name}/{log_dir}'
        os.makedirs(dir2log, exist_ok=True)
        self.writer = SummaryWriter(log_dir=dir2log)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

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

    def info_nce_loss_withmask(self, features, features_lnd):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().gt(0)
        labels = labels.to(self.args.device)
        # keep hard negatives
        # assert hard_neg.shape[0]%4==0
        hard_neg_labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        hard_neg_labels = (hard_neg_labels.unsqueeze(0) == hard_neg_labels.unsqueeze(1)).float()
        hard_neg_labels = hard_neg_labels.to(self.args.device).gt(0)
        hard_neg_1, hard_neg_2, hard_neg_3, hard_neg_4 = torch.split(hard_neg_labels, int(hard_neg_labels.shape[0] / 4))
        hard_neg_mask = ~torch.cat([hard_neg_4, hard_neg_1, hard_neg_2, hard_neg_3], dim=0)

        features = F.normalize(features, dim=1)
        features_lnd = F.normalize(features_lnd, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix_lnd = torch.matmul(features_lnd, features_lnd.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        # discard where landmark feature are more correlated but keep the hard nagative
        threshold_mask = (torch.ones([labels.shape[0],labels.shape[0]])*self.args.threshold).to(self.args.device)
        fea_mask = torch.gt(similarity_matrix_lnd,threshold_mask)
        # fea_mask = torch.nonzero((similarity_matrix_lnd>=self.args.threshold))

        FN_mask = (hard_neg_mask & fea_mask & ~labels)*1
        similarity_matrix[FN_mask==1] = 0
        # TODO: WHAT TO DO WITH THE PICKED OUT SIMILAR ONES
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def info_nce_loss_withmask_agg(self, features, features_lnd):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().gt(0)
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        features_lnd = F.normalize(features_lnd, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix_lnd = torch.matmul(features_lnd, features_lnd.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        similarity_matrix_lnd = similarity_matrix_lnd[~mask].view(similarity_matrix_lnd.shape[0], -1)
        similarity_matrix_lnd = similarity_matrix_lnd[~labels].view(similarity_matrix_lnd.shape[0], -1)
        a = torch.argmax(similarity_matrix_lnd, dim=1)
        labels2 = torch.zeros_like(similarity_matrix_lnd) \
            .scatter_(1, a.unsqueeze(dim=1).to(self.args.device), 1).bool()

        # select and combine multiple positives
        positives1 = similarity_matrix[labels].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives1 = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives2 = negatives1[labels2].view(negatives1.shape[0], -1)
        # select only the negatives the negatives
        negatives2 = negatives1[~labels2].view(negatives1.shape[0], -1)

        logits1 = torch.cat([positives1, negatives2], dim=1)
        logits2 = torch.cat([positives2, negatives2], dim=1)
        labels = torch.zeros(logits1.shape[0], dtype=torch.long).to(self.args.device)

        logits1 = logits1 / self.args.temperature
        logits2 = logits2 / self.args.temperature
        return logits1, logits2, labels

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

        for epoch_counter in range(self.epoch, self.args.epochs):
            for Sample in tqdm(train_loader):
                Anchor = Sample['Anchor']
                Anchor_eye = Sample['Anchor_eye']
                Anchor_mouth = Sample['Anchor_mouth']
                Positive = Sample['Positive']
                Positive_eye = Sample['Positive_eye']
                Positive_mouth = Sample['Positive_mouth']
                NAnchor = Sample['NAnchor']
                NAnchor_eye = Sample['NAnchor_eye']
                NAnchor_mouth = Sample['NAnchor_mouth']
                NPositive = Sample['NPositive']
                NPositive_eye = Sample['NPositive_eye']
                NPositive_mouth = Sample['NPositive_mouth']

                data = torch.cat([Anchor, NAnchor, Positive, NPositive], dim=0).to(self.args.device)
                eye = torch.cat([Anchor_eye, NAnchor_eye, Positive_eye, NPositive_eye], dim=0).to(self.args.device)
                mouth = torch.cat([Anchor_mouth, NAnchor_mouth, Positive_mouth, NPositive_mouth], dim=0).to(self.args.device)
                # Todo: Add masked neg
                # Todo: for cancel FN, no augmentation on face mask
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(x=data)
                    features_e = self.encoder(eye)
                    features_m = self.encoder(mouth)
                    features_lnd = torch.cat([features_e, features_m], dim=1)
                    # logits, labels = self.info_nce_loss_withmask(features, features_lnd)
                    # loss = self.criterion(logits, labels)
                    logits1, logits2, labels = self.info_nce_loss_withmask_agg(features, features_lnd)
                    loss1 = self.criterion(logits1, labels)
                    loss2 = self.criterion(logits2, labels)
                    loss = loss1 + loss2*self.args.negative_lambda

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    top1, top5 = accuracy(logits1, labels, topk=(1, 5))
                    self.writer.add_scalars('Accuracy',
                                            {'acc/top1':top1[0],
                                             'acc/top5':top5[0]}, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            ###############################################################################################
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step(epoch_counter)
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}")

            if epoch_counter%10 == 0 and top1[0] > 90:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
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
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler,
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        logging.info("Training has finished.")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")