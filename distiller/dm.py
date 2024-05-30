import random
from collections import defaultdict
import torch
import numpy as np
# from models.convnet import ConvNet
from models.conv import ConvNet
import time
from tqdm import tqdm
import torch.nn.functional as F
import copy

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)

def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x

class DistributionMatching(object):
    def __init__(self, params):
        self.params = params
        if self.params.data == 'mini_imagenet':
            self.im_size = (3, 84, 84)
        elif self.params.data == 'tiny_imagenet':
            self.im_size = (3, 64, 64)
        else:
            self.im_size = (3, 32, 32)

        self.dsa_params = ParamDiffAug()
        self.global_step = 0


    def distill(self, train_dataset, ipc, num_classes):
        
        images_all = []
        labels_all = []
        images_all = [torch.unsqueeze(train_dataset[i][0], dim=0) for i in range(len(train_dataset))]
        images_all = torch.cat(images_all, dim=0).cuda()
        labels_all = [train_dataset[i][1].item() for i in range(len(train_dataset))]
        label_set = set(labels_all)
        indices_class = defaultdict(list)
        for idx, label in enumerate(labels_all):
            indices_class[label].append(idx)

        def get_images(label, num):
            idx_shuffe = np.random.permutation(indices_class[label])[:num]
            return images_all[idx_shuffe]
        
        # init distilled dataset
        image_syn = torch.randn(size=(num_classes*ipc, self.im_size[0], self.im_size[1], self.im_size[2]), dtype=torch.float, requires_grad=True, device='cuda')
        label_syn = torch.tensor([np.ones(ipc)*i for i in label_set], dtype=torch.long, requires_grad=False, device='cuda').view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        i = 0
        for lab in label_set:
            image_syn.data[i*ipc:(i+1)*ipc] = get_images(lab, ipc).detach().data
            i += 1
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.params.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('ipc', ipc, 'num_classes', num_classes)
        for it in tqdm(range(self.params.distill_iteration)):
            if 'BN' in self.params.inner_model:
                net = ConvNet(channel=3, num_classes=100, net_width=128, net_depth=3, net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=self.im_size[1:]).cuda()
            else:
                net = ConvNet(channel=3, num_classes=100, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=self.im_size[1:]).cuda()

            net.train()

            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0
            if 'BN' in self.params.inner_model:

                ''' update synthetic data '''
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).cuda()
                i = 0
                for c in label_set:
                    batch = self.params.batch
                    img_real = get_images(c, batch)
                    img_syn = image_syn[i*ipc:(i+1)*ipc].reshape(ipc, self.im_size[0], self.im_size[1], self.im_size[2])
                    i += 1
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_params)
                    img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_params)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)
                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)
                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)
                loss += torch.sum((torch.mean(output_real.reshape(num_classes, batch, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, ipc, -1), dim=1))**2)
            
            else:
                loss = torch.tensor(0.0).cuda()
                i = 0
                for c in label_set:
                    
                    total_num = len(indices_class[c])
                    batch = total_num // 2
                    img_real = get_images(c, batch)
                    img_syn = image_syn[i*ipc:(i+1)*ipc].reshape((ipc, self.im_size[0], self.im_size[1], self.im_size[2]))
                    i += 1
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_params)
                    img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_params)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= (num_classes)
            self.params.writer.add_scalar('Loss/Distill', loss_avg, self.global_step)
            self.global_step += 1

        distilled_img = copy.deepcopy(image_syn.detach())
        distilled_label = copy.deepcopy(label_syn.detach())
        return distilled_img, distilled_label