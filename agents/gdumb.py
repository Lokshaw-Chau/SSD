import torch
from torch.utils import data
import math
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, setup_architecture, setup_opt
from utils.utils import maybe_cuda, EarlyStopping
import numpy as np
import random
from tqdm import tqdm


class Gdumb(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Gdumb, self).__init__(model, opt, params)
        self.mem_img = {}
        self.mem_c = {}
        self.early_stopping = EarlyStopping(self.params.min_delta, self.params.patience, self.params.cumulative_delta)
        self.global_step = 0

    def greedy_balancing_update(self, x, y):
        k_c = self.params.mem_size // max(1, len(self.mem_img))
        if y not in self.mem_img or self.mem_c[y] < k_c:
            if sum(self.mem_c.values()) >= self.params.mem_size:
                cls_max = max(self.mem_c.items(), key=lambda k:k[1])[0]
                idx = random.randrange(self.mem_c[cls_max])
                self.mem_img[cls_max].pop(idx)
                self.mem_c[cls_max] -= 1
            if y not in self.mem_img:
                self.mem_img[y] = []
                self.mem_c[y] = 0
            self.mem_img[y].append(x)
            self.mem_c[y] += 1

    def train_learner(self, x_train, y_train, labels):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        print('Updating Buffer...')
        for i, batch_data in enumerate(train_loader):
            # batch update
            batch_x, batch_y = batch_data
            batch_x = maybe_cuda(batch_x, self.cuda)
            batch_y = maybe_cuda(batch_y, self.cuda)
            # update mem
            for j in range(len(batch_x)):
                self.greedy_balancing_update(batch_x[j], batch_y[j].item())
        self.early_stopping.reset()
        print('Train with Buffer...')
        self.train_mem()
        self.after_train()

    def train_mem(self):
        mem_x = []
        mem_y = []
        for i in self.mem_img.keys():
            mem_x += self.mem_img[i]
            mem_y += [i] * self.mem_c[i]

        mem_x = torch.stack(mem_x)
        mem_y = torch.LongTensor(mem_y)
        self.model = setup_architecture(self.params)
        self.model = maybe_cuda(self.model, self.cuda)
        # lr = self.params.learning_rate
        opt = setup_opt(self.params.optimizer, self.model, self.params.learning_rate, self.params.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=self.params.minlr)

        loss = math.inf
        for i in tqdm(range(self.params.mem_epoch)):
            idx = np.random.permutation(len(mem_x)).tolist()
            mem_x = maybe_cuda(mem_x[idx], self.cuda)
            mem_y = maybe_cuda(mem_y[idx], self.cuda)
            self.model = self.model.train()
            batch_size = 200
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
            # if i in lr_schedule:
            #     lr *= 0.1
            #     opt = setup_opt(self.params.optimizer, self.model, lr, self.params.weight_decay)

