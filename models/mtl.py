# -*- coding: utf-8 -*-
# @Time    : 2021/1/31 17:15
# @Author  : -----↖(^?^)↗
# @FileName: mlp.py
# -------------------------start-----------------------------

import torch.nn as nn
import torch


class MTL_MLP(nn.Module):
    def __init__(self, out_fea=1, hidden=[128, 100, 100, 100], ratio=0.2, brp=False,
                 fixed=False, lstm=False, input_size=27,
                 hidden_size=100, num_layers=1, bidir=False):
        super(MTL_MLP, self).__init__()
        #  FC
        self.hidden = hidden
        self.brp = brp
        self.fixed = fixed
        self.out_fea = out_fea
        self.ratio = ratio
        self.layer = len(self.hidden) -1
        self.mlp = nn.ModuleList([nn.Linear(self.hidden[i], self.hidden[i+1]) for i in range(self.layer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.hidden[i+1]) for i in range(self.layer)])
        self.relu = nn.ModuleList([nn.ReLU() for i in range(self.layer)])
        # lstm
        self.input_size = input_size
        self.orig_hidden = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = lstm
        self.bidir = bidir
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidir, batch_first=True)
        if self.bidir:
            self.hidden_size = self.hidden_size * 2
        if self.num_layers > 1:
            self.hidden_size = self.hidden_size * self.num_layers

        # final classifier
        if self.lstm:
            fc_in = self.hidden_size
        else:
            fc_in = self.hidden[-1]

        if self.fixed:
            for p in self.parameters():
                p.requires_grad = False

        out_dim = 2
        self.final_act = nn.LogSoftmax(dim=1)

        self.fc1 = nn.Linear(fc_in * 2, out_dim)
        self.fc2 = nn.Linear(fc_in * 2, out_dim)
        self.fc3 = nn.Linear(fc_in * 2, out_dim)
        self.fc4 = nn.Linear(fc_in * 2, out_dim)
        self.fc_final = nn.Linear(fc_in, 1)
        self.dp = nn.ModuleList([nn.Dropout(self.ratio) for i in range(self.layer)])
        fcs = []
        dps = []
        act_fuc = []

        for _ in range(20):
            fcs.append(nn.Linear(self.orig_hidden, 1))
        self.fcs = nn.ModuleList(fcs)
        self.dps = nn.ModuleList(dps)
        self.act_fuc = nn.ModuleList(act_fuc)

    def forward_one(self, x):
        for i in range(self.layer):
            x = self.relu[i](self.bn[i](self.mlp[i](x)))
            x = self.dp[i](x)
        return x

    def lstm_one(self, x):
        self.lstm_layer.flatten_parameters()
        x, h_c = self.lstm_layer(x)
        return h_c[0]

    def regress(self, x):
        x, h_c = self.lstm_layer(x)
        b, s, h = x.shape
        for _ in range(s):
            if _ == 0:
                latency = self.fcs[_](x[:, _, :])
            else:
                latency += self.fcs[_](x[:, _, :])
        return latency



    def forward(self, x):
        if self.brp not in [True]:
            if self.lstm:
                x = self.regress(x)
                return x
            else:
                x = self.forward_one(x)
                x = self.fc_final(x)
                return x
        else:
            if self.lstm:
                x1 = self.lstm_one(x[:, 0])
                x2 = self.lstm_one(x[:, 1])
                if self.bidir or self.num_layers > 1:
                    x1 = x1.transpose(1, 0).reshape(-1, self.hidden_size)
                    x2 = x2.transpose(1, 0).reshape(-1, self.hidden_size)
                else:
                    x1 = x1.reshape(-1, self.hidden_size)
                    x2 = x2.reshape(-1, self.hidden_size)
                x = torch.cat([x1, x2], dim=-1)

            else:
                x1 = self.forward_one(x[:, 0])
                x2 = self.forward_one(x[:, 1])
                x = torch.cat([x1, x2], dim=1)

            x1 = self.fc1(x)
            x2 = self.fc2(x)
            x3 = self.fc3(x)
            x4 = self.fc4(x)

            x1 = self.final_act(x1)
            x2 = self.final_act(x2)
            x3 = self.final_act(x3)
            x4 = self.final_act(x4)
            return torch.stack([x1, x2, x3, x4], 1)
