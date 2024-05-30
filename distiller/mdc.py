import random
from collections import defaultdict
import torch
import numpy as np
# from models.convnet import ConvNet
from models.convnet_mdc import ConvNet
import time
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from math import ceil
import copy
from distiller.reg_ipcx import Regularizer, feat_loss_for_ipc_reg, grad_loss_for_img_update
from dm import DiffAugment

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device
        # self.dream = args.slct_type == 'dream' # whether to use DREAM initialization

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)

        print(f"\nDefine synthetic data: {self.data.shape}")

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, img, label):
        """Condensed data initialization
        """
        self.data.data[self.ipc * label:self.ipc * (label + 1)] = img.detach().data.to(self.device)

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            indices = indices[:max_size]
            data = data[indices]
            target = target[indices]
            # [49, 123, 231, ...]
        else:
            indices = np.arange(data.shape[0])
            # [ 0, ..., 39]

        return data, target, indices

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target, indices = self.subsample(data, target, max_size=max_size)
        return data, target, indices

    def loader(self, args, augment=True, ipcx=-1, indices=None):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            if ipcx > 0:
                idx_to = self.ipc * c + ipcx
            else:
                idx_to = self.ipc * (c + 1)
            
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            if indices is not None: # use indices after decoding
                data = data[indices]
                target = target[indices]

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print(f"Decode condensed data: {data_dec.shape}")
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, ipcx=-1, indices=None, aim_run=None, step=None, context=None):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment, ipcx=ipcx, indices=indices)
        return test_data(args, loader, val_loader, logger=logger, aim_run=aim_run, step=step, context=context)


class MultisizeDatasetCondensation(object):
    def __init__(self, params):
        self.params = params
        if self.params.data == 'mini_imagenet':
            self.im_size = (3, 84, 84)
        elif self.params.data == 'tiny_imagenet':
            self.im_size = (3, 64, 64)
        else:
            self.im_size = (3, 32, 32)

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
        synset = Synthesizer(self.params, num_classes, self.im_size[0], self.im_size[1], self.im_size[2])
        label_syn = torch.tensor([np.ones(ipc)*i for i in label_set], dtype=torch.long, requires_grad=False, device='cuda').view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        i = 0
        for lab in label_set:
            #image_syn.data[i*ipc:(i+1)*ipc] = get_images(lab, ipc).detach().data
            synset.init(get_images(lab, ipc), i)
            i += 1
        # augmentation


        # MDC setup
        regularizer_list = []
        for c in range(num_classes):
            regularizer_list.append(Regularizer(ipc))
        freeze_ipc = -1

        # compute regularize index
        ipcx_index_class_dict = {}
        ipcs_list = list(range(1, ipc))
        self.params.adaptive_reg_list = ipcs_list
        for ipcx in ipcs_list:
            ipcx_num = ipcx * self.params.factor ** 2
            ipcx_index_class = [i for i in range(ipcx_num)]
            ipcx_index_class_dict[ipcx] = ipcx_index_class
        
        optim_img = torch.optim.SGD(synset.parameters(), lr=5e-3, momentum=0.5)
        # Data Distillation
        n_iter = self.params.distill_iteration * 100 // self.params.inner_loop
        for it in tqdm(n_iter):
            net = ConvNet(channel=3, num_classes=100, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', im_size=self.im_size[1:]).cuda()
            net.train()
            optim_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            loss_total = 0
            synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

            loss_list_per_class = [[] for i in range(num_classes)] # should have shape (nclass, )
            freeze_ipc_list = [-1 for i in range(num_classes)]

            for model_loop in range(self.params.inner_loop):
                for c in range(len(label_set)):
                    img_real = get_images(label_set[c], ipc)
                    img_syn, lab_syn, sampled_indices = synset.sample(c, max_size=256)
                    regularizer_list[c].update_status(it)   # update regularizer status
                    freeze_ipc_list[c] = regularizer_list[c].get_freeze_ipc()

                    if freeze_ipc_list[c] > 0: # freeze the ipcx
                        freeze_ipc_idx = ipcx_index_class_dict[freeze_ipc_list[c]]
                        detached_img_syn = img_syn[freeze_ipc_idx].detach()
                        detached_img_syn.requires_grad = False

                        # replace the freeze ipcx with the detached version
                        img_syn[freeze_ipc_idx] = detached_img_syn
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_params)
                img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=self.dsa_params)
                if (it == 0) or ((it + 1) % self.params.adaptive_period == 0):
                    # loss_list contains the loss of all ipcx in search space
                    loss_list = feat_loss_for_ipc_reg(self.params, img_real, img_syn, net, indices=ipcx_index_class_dict)
                    loss_list_per_class[c].append(loss_list)
                    reg_ipcx_list = regularizer_list[c].get_regularized_ipc()
                    loss = grad_loss_for_img_update(self.params, img_real, img_syn, lab, lab_syn, net, ipcx_list=reg_ipcx_list, indices=ipcx_index_class_dict)

                loss_total += loss.item()

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()

            for _ in range(1):
                    top1, top5, model_loss = train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net)


        return distilled_img, distilled_label