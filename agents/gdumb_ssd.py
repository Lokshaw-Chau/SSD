import torch
from torch.utils import data
import math
from agents.base import ContinualLearner
from utils.buffer.buffer import Buffer, DynamicBuffer
from continuum.data_utils import dataset_transform, BalancedSampler
from utils.setup_elements import transforms_match, input_size_match
from utils.setup_elements import transforms_match, setup_architecture, setup_opt
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from utils.utils import maybe_cuda, EarlyStopping
import numpy as np
import random
import torch.nn as nn


class GdumbSSD(ContinualLearner):
    def __init__(self, model, opt, params):
        super(GdumbSSD, self).__init__(model, opt, params)
        self.buffer = DynamicBuffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.queue_size = params.queue_size
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )

    def train_learner(self, x_train, y_train, labels):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_sampler = BalancedSampler(x_train, y_train, self.batch)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, num_workers=0,
                                       drop_last=True, sampler=train_sampler)
        self.buffer.new_condense_task(labels)

        aff_x = []
        aff_y = []
        step=0
        for i, batch_data in enumerate(train_loader):
            # batch update
            batch_x, batch_y = batch_data
            batch_x = maybe_cuda(batch_x, self.cuda)
            batch_y = maybe_cuda(batch_y, self.cuda)
            
            aff_x.append(batch_x)
            aff_y.append(batch_y)
            if len(aff_x) > self.queue_size:
                aff_x.pop(0)
                aff_y.pop(0)
            # update mem
            loss = self.buffer.update(batch_x, batch_y, aff_x=aff_x, aff_y=aff_y, update_index=i, transform=self.transform)
            if loss is not None:
                self.params.logger.add_scalar('img_loss', loss, step)
                step += 1
        #self.early_stopping.reset()
        self.train_mem()
        self.after_train()

    def train_mem(self):
        mem_x = self.buffer.buffer_img
        mem_y = self.buffer.buffer_label
        self.model = setup_architecture(self.params)
        self.model = maybe_cuda(self.model, self.cuda)
        opt = setup_opt(self.params.optimizer, self.model, self.params.learning_rate, self.params.weight_decay)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=self.params.minlr)

        #loss = math.inf
        for i in range(self.params.mem_epoch):
            idx = np.random.permutation(len(mem_x)).tolist()
            mem_x = maybe_cuda(mem_x[idx], self.cuda)
            mem_y = maybe_cuda(mem_y[idx], self.cuda)
            self.model = self.model.train()
            batch_size = self.params.batch
            #scheduler.step()
            #if opt.param_groups[0]['lr'] == self.params.learning_rate:
            #    if self.early_stopping.step(-loss):
            #        return
            for j in range(len(mem_y) // batch_size):
                opt.zero_grad()
                logits = self.model.forward(mem_x[batch_size * j:batch_size * (j + 1)])
                loss = self.criterion(logits, mem_y[batch_size * j:batch_size * (j + 1)])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip)
                opt.step()

