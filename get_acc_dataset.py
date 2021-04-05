# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 10:53
# @Author  : -----↖(^?^)↗
# @FileName: get_acc_dataset.py
# -------------------------start-----------------------------

from ofa.tutorial.accuracy_predictor import AccuracyPredictor
import csv
import os
import torch
from ofa.model_zoo import ofa_net
from ofa.utils.pytorch_utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
acc_pred = AccuracyPredictor()
path = './latencydataset/latency.csv'
csv_f = open(path, 'r')
reader = csv.reader(csv_f)
new = './latencydataset/with_pred_acc.csv'
wri_f = open(new,'w')
writer = csv.writer(wri_f)
writer.writerow(['arch_config','gpu latency','pred_accs','flops','params'])
all_dict = []

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')
for i, line in enumerate(reader):
    if i>0:
        dic = eval(line[0])
        gpu_lat = eval(line[-3])
        # ks, e, d, r = dic['ks'], dic['e'], dic['d'], dic['r'][0]
        ofa_network.set_active_subnet(ks=dic['ks'], d=dic['d'], e=dic['e'])
        subnet = ofa_network.get_active_subnet().to(device)

        flops = count_net_flops(subnet, data_shape=(1, 3, dic['r'][0], dic['r'][0]))
        params = count_parameters(subnet)
        print(i, flops,params)
        acc = acc_pred.predict_accuracy([dic])
        writer.writerow([dic, gpu_lat, round(acc.item(),4),flops,params])


