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

class GDumbDD(ContinualLearner):
    def __init__(self, model, opt, params):
        super(GDumbDD, self).__init__(model, opt, params)
        #self.buffer = Buffer(model, params)
        self.mem_img = None
        self.mem_c = None
        self.distiller = DistributionMatching(params)
        self.early_stopping = EarlyStopping(self.params.min_delta, self.params.patience, self.params.cumulative_delta)
        self.global_step = 0
        self.task_cnt = 0

    def train_learner(self, x_train, y_train, labels):
        self.before_train(x_train, y_train)
        # if self.mem_img is not None:
        #     mem = self.mem_img.permute(0, 2, 3, 1).cpu().numpy()
        #     c = self.mem_c.cpu().numpy()
        #     x_train = np.concatenate([mem, x_train], axis=0)
        #     y_train = np.concatenate([c, y_train], axis=0)
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        print('Distilling Task Dataset...')
        # distill task data
        self.task_cnt += 1
        # ipc = self.params.mem_size // (self.task_cnt * self.params.num_classes_per_task)
        ipc = 20
        num_classes = self.params.num_classes_per_task
        img_syn, lab_syn = self.distiller.distill(train_dataset, ipc, num_classes)
        # update buffer
        if self.mem_img is None:
            self.mem_img = img_syn
            self.mem_c = lab_syn
        else:
            self.mem_img = torch.cat([self.mem_img, img_syn], dim=0)
            self.mem_c = torch.cat([self.mem_c, lab_syn], dim=0)
        # train model
        self.early_stopping.reset()
        # set up model
        print("Training with Buffer...")
        self.train_mem()
        self.after_train()

    def train_mem(self):

        mem_x, mem_y = self.mem_img, self.mem_c
        print('Training with %d samples' % len(mem_x))

        self.model = setup_architecture(self.params)
        self.model = maybe_cuda(self.model, self.cuda)
        opt = setup_opt(self.params.optimizer, self.model, self.params.learning_rate, self.params.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=self.params.minlr)

        loss = math.inf
        for i in tqdm(range(self.params.mem_epoch)):
            idx = np.random.permutation(len(mem_x)).tolist()
            mem_x = maybe_cuda(mem_x[idx], self.cuda)
            mem_y = maybe_cuda(mem_y[idx], self.cuda)
            self.model = self.model.train()
            
            #batch_size = self.params.batch
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