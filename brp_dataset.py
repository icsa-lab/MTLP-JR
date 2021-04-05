# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 15:30
# @Author  : -----↖(^?^)↗
# @FileName: dataset.py
# -------------------------start-----------------------------
import copy
from utils import construct_maps
from torch.utils.data import TensorDataset, DataLoader
import torch
import random
import pandas as pd


ks_map = construct_maps(keys=(3, 5, 7))
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))
reso_map = construct_maps(keys=(160, 176, 192, 208, 224))
channel_map = construct_maps(keys=(24, 32, 48, 96, 136, 192))
stride_map = construct_maps(keys=(1, 2))


class Dataprocess():
    def __init__(self, file_path):
        self.file_path = file_path

    def encode(self, lstm=True):
        all_feats = []
        gpu, pred_accs, flops, params = [], [], [], []
        top1 = []
        df = pd.read_csv(self.file_path, sep='\t')
        population = df.values

        for i, sample in enumerate(population):
            sample = eval(sample[0])
            para = sample[0]
            ks_list = copy.deepcopy(para['ks'])
            # ks_list = [0] * len(ks_list)
            ex_list = copy.deepcopy(para['e'])
            d_list = copy.deepcopy(para['d'])
            r = copy.deepcopy(para['r'])[0]
            feats = self.spec2feats_all(ks_list, ex_list, d_list, r, lstm)
            gpu.append(sample[1])
            pred_accs.append(sample[2])
            flops.append(sample[3])
            params.append(sample[4])
            top1.append(sample[5])
            all_feats.append(feats)
        if lstm:
            all_feats = torch.stack(all_feats)
        else:
            all_feats = torch.cat(all_feats, 0).reshape(-1, len(all_feats[0]))  # row
        gpu = torch.tensor(gpu).reshape(len(gpu), -1)
        pred_accs = torch.tensor(pred_accs).reshape(len(pred_accs), -1)
        flops = torch.tensor(flops).reshape(len(flops), -1)
        params = torch.tensor(params).reshape(len(params), -1)
        top1 = torch.tensor(top1).reshape(len(top1), -1)
        return all_feats, gpu, pred_accs, flops, params, top1


    @staticmethod
    def spec2feats_all(ks_list, ex_list, d_list, r, lstm):
        channels = [24, 32, 48, 96, 136, 192]
        strides = [1, 2, 2, 2, 1, 2]
        se = [0, 0, 1, 0, 1, 1]
        start = 0
        end = 4

        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4
        tmp = []

        for i, (ks, ex) in enumerate(zip(ks_list, ex_list)):
            ks_ten, ex_ten = [0, 0, 0], [0, 0, 0]
            reso = [0 for _ in range(5)]
            inchannel = [0 for _ in range(6)]
            outchannel = [0 for _ in range(6)]
            stride = [0 for _ in range(2)]
            shortcut = 0
            if ks != 0:
                if i == 0:
                    in_channel = 24
                    out_channel = 32
                    ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]], inchannel[channel_map[in_channel]], \
                    outchannel[channel_map[out_channel]] = 1, 1, 1, 1, 1
                    tmp.append(torch.cat(
                        [torch.tensor(a) for a in [ks_ten, ex_ten, reso, inchannel, outchannel, [0, 1], [0, 0]]]))
                else:
                    if i % 4 == 0:
                        in_channel = channels[i // 4]
                        out_channel = channels[i // 4 + 1]
                        now_stride = strides[i // 4]
                    else:
                        in_channel = channels[i // 4 + 1]
                        out_channel = channels[i // 4 + 1]
                        now_stride = 1
                        shortcut = 1
                    ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]], inchannel[channel_map[in_channel]], \
                    outchannel[channel_map[out_channel]] = 1, 1, 1, 1, 1
                    stride[stride_map[now_stride]] = 1
                    tmp.append(torch.cat([torch.tensor(a) for a in [ks_ten, ex_ten, reso, inchannel, outchannel, stride,
                                                                    [se[i // 4], shortcut]]]))
            else:
                tmp.append(torch.tensor([0] * 27))
        if lstm:

            ten = torch.stack(tmp)
        else:
            ten = torch.cat(tmp, -1)
        return ten


class ProductList():
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        i1, i2 = self.unmerge(idx)
        return self.values[i1], self.values[i2]

    def __len__(self):
        return len(self.values) * (len(self.values) - 1)

    def unmerge(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError()

        i1 = idx // (len(self.values) - 1)
        i2 = idx % (len(self.values) - 1)
        if i1 <= i2:
            i2 = (i2 + 1) % len(self.values)
        return i1, i2


def normalization(targets):
    max = torch.max(targets)
    min = torch.min(targets)
    gap = max - min
    targets = (targets - min).float() / gap
    return targets


def soft_max(data_pair):
    targets = []
    for pair in data_pair:
        tmp = torch.softmax(torch.stack([torch.tensor(pair[0][1]), torch.tensor(pair[1][1])], 1), 1)
        targets.append(tmp)

    return targets


def data_loader_all(num, bs, file, lstm=True, brp=True):
    dataprocess = Dataprocess(file)
    # all_feats, gpu, pred_accs, flops, params, top1
    all_feats, gpu, pred_accs, flops, params, top1 = dataprocess.encode(lstm=lstm)
    if not brp:
        data_set = TensorDataset(all_feats[:num], gpu[:num])
        orig_data = None
    elif brp in [True]:
        gpu = normalization(gpu)
        top1 = normalization(top1)
        params = normalization(params)
        flops = normalization(flops)
        targets = [(gpu[i], top1[i], flops[i], params[i]) for i in range(len(all_feats))]
        data = [[i, j] for i, j in zip(all_feats, targets)]
        random.seed(888)
        random.shuffle(data)
        data = data[:num]
        orig_data = data
        data_pair = ProductList(data)
        changed_tar = soft_max(data_pair)
        data_set = [(torch.stack([i[0][0], i[1][0]]), j) for i, j in zip(data_pair, changed_tar)]

    data_loader = DataLoader(dataset=data_set, batch_size=bs, shuffle=False, drop_last=False,
                             pin_memory=True, num_workers=8)
    return data_loader, orig_data

