import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer, DynamicBuffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform, BalancedSampler
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from utils.utils import maybe_cuda, EarlyStopping
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from utils.setup_elements import transforms_match, setup_architecture, setup_opt
import math
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from distiller.dm import DistributionMatching


class SummarizeStreamData(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SummarizeStreamData, self).__init__(model, opt, params)
        self.buffer = DynamicBuffer(model, params)
        self.mem_size = params.mem_size
        self.distiller = DistributionMatching(params)
        self.early_stopping = EarlyStopping(self.params.min_delta, self.params.patience, self.params.cumulative_delta)
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )
        self.global_step = 0
        self.distill_step = 0

    def train_learner(self, x_train, y_train, labels):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                       drop_last=True, sampler=train_sampler)
        
        self.buffer.new_condense_task(labels)
        print('Reservoir Updating Buffer...')
        for i, batch_data in enumerate(train_loader):
            # batch update
            batch_x, batch_y = batch_data
            batch_x = maybe_cuda(batch_x, self.cuda)
            batch_y = maybe_cuda(batch_y, self.cuda)
            # update mem
            self.buffer.update(batch_x, batch_y, condense_flag=False, transform=self.transform)
        
        print('Distilling Dataset into Buffer...')
        img_syn, label_syn = self.distiller.distill(train_dataset, 20, 20)
        self.buffer.update(img_syn, label_syn, condense_flag=True, transform=self.transform)
        # for ep in tqdm(range(self.epoch)):
        #     for i, batch_data in enumerate(train_loader):
        #         # batch update
        #         batch_x, batch_y = batch_data
        #         batch_x = maybe_cuda(batch_x, self.cuda)
        #         batch_y = maybe_cuda(batch_y, self.cuda)
        #         # update mem
        #         distill_loss = self.buffer.update(batch_x, batch_y, condense_flag=True, transform=self.transform)
        #         self.params.writer.add_scalar('Loss/Distill', distill_loss, self.distill_step)
        #         self.distill_step += 1
        # train model
        self.early_stopping.reset()
        # set up model
        print("Training with Buffer...")
        self.train_mem()
        self.after_train()

    def train_mem(self):

        mem_x, mem_y = self.buffer.buffer_img, self.buffer.buffer_label

        self.model = setup_architecture(self.params)
        self.model = maybe_cuda(self.model, self.cuda)
        opt = setup_opt(self.params.optimizer, self.model, self.params.learning_rate, self.params.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=self.params.minlr)

        loss = math.inf
        for i in range(self.params.mem_epoch):
            idx = np.random.permutation(len(mem_x)).tolist()
            mem_x = maybe_cuda(mem_x[idx], self.cuda)
            mem_y = maybe_cuda(mem_y[idx], self.cuda)
            self.model = self.model.train()
            batch_size = self.params.batch
            scheduler.step()
            if opt.param_groups[0]['lr'] == self.params.learning_rate:
               if self.early_stopping.step(-loss):
                   return
            for j in range(len(mem_y) // batch_size):
                opt.zero_grad()
                logits = self.model.forward(mem_x[batch_size * j:batch_size * (j + 1)])
                loss = self.criterion(logits, mem_y[batch_size * j:batch_size * (j + 1)])
                loss.backward()
                self.params.writer.add_scalar('Loss/train', loss, self.global_step)
                self.params.writer.add_scalar('LR/train', opt.param_groups[0]['lr'], self.global_step)
                self.global_step += 1

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)
                opt.step()