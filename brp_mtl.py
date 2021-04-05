# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 9:52
# @Author  : -----↖(^?^)↗
# @FileName: brp_run.py
# -------------------------start-----------------------------

from brp_dataset import data_loader_all
import time
from utils import *
from models.mtl import MTL_MLP
import argparse
import torch
import torch.nn as nn
import functools
import copy
import os
from sklearn.metrics import mean_squared_error, r2_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='trainging epochs', type=int, default=100)
parser.add_argument('--bs', type=list, default=[256, 2048, 2048])
parser.add_argument('--lr', type=float, default=3.5e-3)
parser.add_argument('--ratio', help='the ratio of train_data and valid_data', nargs='+', type=int, default=[50, 50, 50])
parser.add_argument('--train_file', type=str, default='./data/train.csv')
parser.add_argument('--val_file', type=str, default='./data/valid.csv')
parser.add_argument('--te_file', type=str, default='./data/test.csv')
parser.add_argument('--input_size', type=int, default=27)
parser.add_argument('--bidir', type=bool, default=False)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--root_log_path', type=str, default='./logs/new')
parser.add_argument('--mtl', type=bool, default=True)
parser.add_argument('--brp', default=True)
parser.add_argument('--lr_scheduler', default='cosine')
parser.add_argument('--lr_patience', default=10)
parser.add_argument('--weight_decay', default=5.0e-4)
parser.add_argument('--hidden', type=list, default=[540, 200, 200, 400])
parser.add_argument('--dp_ratio', type=float, default=0.2)
parser.add_argument('--es_patience', type=int, default=20)
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--mtl_ratio', type=list, default=[1, 0, 0, 0])
parser.add_argument('--tl', type=bool, default=False)
parser.add_argument('--fixed', type=bool, default=False)
parser.add_argument('--lstm', type=bool, default=True)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--tl_lr', type=float, default=3.5e-5)
parser.add_argument('--tl_weight_decay', type=float, default=5.0e-4)
parser.add_argument('--regress_results', type=str, default='./regress')
parser.add_argument('--brp_results', type=str, default='./brp_results')
parser.add_argument('--comment', type=str, default='no r')
args = parser.parse_args()
if not os.path.exists(args.root_log_path):
    os.mkdir(args.root_log_path)

time_ = time.strftime('%Y%m%d%H%M%S%S')
working = args.root_log_path + '/{}'.format(time_)
if not os.path.exists(working):
    os.mkdir(working)
handler = SummaryWriter(log_dir=working, )
handler.add_text('args', str(args.__dict__))

if not args.brp:
    criterion = torch.nn.MSELoss()
else:
    criterion = torch.nn.KLDivLoss()


def train_val(net, epochs, train_loader, valid_loader):
    if args.tl:
        net.load_state_dict(torch.load(args.model_path))
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.tl_lr, weight_decay=args.tl_weight_decay)
        if args.fixed:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.tl_lr,
                                          weight_decay=args.tl_weight_decay)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=args.lr_patience,
                                                               threshold=0.01, verbose=True)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    else:
        raise ValueError(f'Unknown lr scheduler: {args.lr_scheduler}')

    es = EarlyStopping(mode='max', patience=args.es_patience)

    best_epochs = 0
    max_val_acc = [0, 0, 0, 0]
    now_train_loss, now_val_loss = [], []
    best_model = None
    ite = 0
    min_val_mse = 1000000000
    for j in range(epochs):
        train_loss_all = []
        train_loss = []
        net.train()
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = net(inputs)
            loss = torch.stack([criterion(outputs[:, i], labels[:, i]) for i in range(outputs.shape[1])])
            loss_all = sum([args.mtl_ratio[i] * loss[i] for i in range(outputs.shape[1])])
            train_loss_all.append(loss_all.item())
            train_loss.append(loss)
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            ite += 1

        val_loss_sum, val_loss, val_acc = val(net, valid_loader)
        train_loss = torch.mean(torch.stack(train_loss), 0)
        train_loss_all = np.mean(train_loss_all)
        print(f'epochs:{j}', end=' ')
        print('tr_loss_all:{}-train_loss:{}-val_loss_all:{}-val_loss:{}-val_acc:{}'.
              format(train_loss_all, train_loss.data, val_loss_sum, val_loss.data, val_acc))
        if not args.brp:
            if val_loss_sum < min_val_mse:
                best_model = copy.deepcopy(net)
                best_epochs = j
                min_val_mse = val_loss_sum
                now_train_loss = train_loss
                now_val_loss = val_loss
            handler.add_scalar('train_loss/loss', train_loss_all, j)
            handler.add_scalar('val_loss/loss', val_loss_sum, j)
        else:
            handler.add_scalar('train_loss/loss', train_loss_all, j)
            handler.add_scalar('train_loss/loss0', train_loss[0], j)
            handler.add_scalar('train_loss/loss1', train_loss[1], j)
            handler.add_scalar('train_loss/loss2', train_loss[2], j)
            handler.add_scalar('train_loss/loss3', train_loss[3], j)
            handler.add_scalar('val_loss/loss', val_loss_sum, j)
            handler.add_scalar('val_loss/loss0', val_loss[0], j)
            handler.add_scalar('val_loss/loss1', val_loss[1], j)
            handler.add_scalar('val_loss/loss2', val_loss[2], j)
            handler.add_scalar('val_loss/loss3', val_loss[3], j)
            handler.add_scalar('val_acc/acc0', val_acc[0], j)
            handler.add_scalar('val_acc/acc1', val_acc[1], j)
            handler.add_scalar('val_acc/acc2', val_acc[2], j)
            handler.add_scalar('val_acc/acc3', val_acc[3], j)

            index = args.index
            if val_acc[index] > max_val_acc[index]:
                max_val_acc = val_acc
                best_epochs = j
                best_model = copy.deepcopy(net)
                now_train_loss = train_loss
                now_val_loss = val_loss

            if args.lr_scheduler == 'plateau':
                if j > 20:
                    scheduler.step(val_loss[index])
            else:
                scheduler.step()

            if j > 20:
                if es.step(val_acc[index]):
                    torch.save(best_model.state_dict(), os.path.join(working, 'model-{}.pkl'.format(max_val_acc)))
                    print('Early stopping criterion is met, stop training now')
                    break

    if not args.brp:
        torch.save(best_model.state_dict(), os.path.join(working, 'model.pkl'))
        print(f'min_val_mse:{min_val_mse}')
        print(f'now_train_loss:{now_train_loss}')
        print(f'now_val_loss:{now_val_loss}')
        print(f'current epoch:{best_epochs}')
        return best_model, (min_val_mse, now_train_loss, now_val_loss, best_epochs)
    else:
        torch.save(best_model.state_dict(), os.path.join(working, 'model.pkl'))
        print(f'max valid acc:{max_val_acc}')
        print(f'current val loss:{now_val_loss.data}')
        print(f'current train loss:{now_train_loss.data}')
        print(f'current epochs:{best_epochs}')
        return best_model, (max_val_acc, now_train_loss.data, now_val_loss.data, best_epochs)


def val(net, valid_loader):
    val_loss_all = []
    val_loss = []
    label_ls, out_ls = [], []
    net.eval()
    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader)):
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = net(inputs)
            loss = torch.stack([criterion(outputs[:, i], labels[:, i]) for i in range(outputs.shape[1])])
            loss_all = sum([args.mtl_ratio[i] * loss[i] for i in range(outputs.shape[1])])
            val_loss_all.append(loss_all.item())
            val_loss.append(loss)
            label_ls.append(labels)
            out_ls.append(outputs)
        label_ls, out_ls = torch.cat(label_ls), torch.cat(out_ls)
        val_acc = []
        if args.brp:
            for i in range(4):
                val_acc.append(corr_acc(out_ls[:, i], label_ls[:, i]))

        val_loss_all = np.mean(val_loss_all)
        val_loss = torch.mean(torch.stack(val_loss), 0)
        return val_loss_all, val_loss, val_acc


def corr_acc(out_ls, label_ls):
    lat_corr = 0
    for result, latencies in zip(out_ls, label_ls):
        rv1, rv2 = result[0].cpu().item(), result[1].cpu().item()
        if latencies[0] > latencies[1]:
            if rv1 > rv2:
                lat_corr += 1
        elif rv1 < rv2:
            lat_corr += 1

    val_acc = lat_corr / len(out_ls)
    return val_acc


def test(model, test_loader, orig_data):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0

        def predictor_compare(v1, v2):
            nonlocal total
            nonlocal correct
            total += 1
            gs = [v1[0], v2[0]]
            gs = torch.stack(gs, 0)
            gs = torch.unsqueeze(gs, dim=0).to(device, dtype=torch.float)
            latencies = torch.stack([v1[1][args.index], v2[1][args.index]], 0).to(device)
            result = model(gs)
            rv1, rv2 = result[0][args.index][0].cpu().item(), result[0][args.index][1].cpu().item()
            if latencies[0] > latencies[1]:
                if rv1 > rv2:
                    correct += 1
            elif rv1 < rv2:
                correct += 1
            # we want higher number to appear later (have higher "score"), so (v1 - v2) should get us the correct order
            return rv1 - rv2

        labels = []
        results = []
        print('begin testing....')
        for data in tqdm(test_loader, total=len(test_loader)):
            input, label = data
            input = input.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            result = model(input)
            results.append(result)
            labels.append(label)

        labels = torch.cat(labels)
        results = torch.cat(results)
        test_acc = []
        if not args.brp:
            mse = mean_squared_error(labels.cpu(), results.cpu())
            lee_way = leeway(results.cpu(), labels.cpu())
            print(f'lee_way:{lee_way}')
            csv_file = open(working + '/results.csv', 'w')
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow(['results', 'labels'])
            for i in range(len(results)):
                csvwriter.writerow([results[i].cpu().numpy()[0], labels[i].cpu().numpy()[0]])
            figure = draw(results.cpu(), labels.cpu(), working + '/scatter.jpg')
            handler.add_figure('figure', figure=figure)
            return mse, lee_way

        else:
            sorted_values = sorted(orig_data, key=functools.cmp_to_key(predictor_compare))
            sorted_values = {feat: (gt, idx) for idx, (feat, gt) in enumerate(sorted_values)}
            gt_rank = sorted(sorted_values.values(), key=lambda x: x[0][args.index])

            gt_figure = draw_rank(list(range(len(gt_rank))), [i[1] for i in gt_rank], path=working + '/rank.jpg')
            handler.add_figure('gt_figure', gt_figure)
            rank_file = open(working + '/rank.csv', 'w')
            rank_writer = csv.writer(rank_file)
            rank_writer.writerow(['gt rank', 'pre rank', 'gt'])
            r2 = r2_score(list(range(len(gt_rank))), [i[1] for i in gt_rank])
            print(f'r2:{r2}')
            for i, (gt, idx) in enumerate(gt_rank):
                rank_writer.writerow([i, idx, gt])
            for i in range(4):
                test_acc.append(corr_acc(results[:, i], labels[:, i]))
            print(f'test accuracy:{test_acc}')
            return test_acc, None


def main():
    net = MTL_MLP(out_fea=1, hidden=args.hidden, ratio=args.dp_ratio, brp=args.brp,
                  fixed=args.fixed,lstm=args.lstm, input_size=args.input_size,
                  hidden_size=args.hidden_size, num_layers=args.num_layers,
                  bidir=args.bidir).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    train_loader, _ = data_loader_all(num=args.ratio[0], bs=args.bs[0], file=args.train_file, lstm=args.lstm, brp=args.brp)
    valid_loader, _ = data_loader_all(num=args.ratio[1], bs=args.bs[1], file=args.val_file,  lstm=args.lstm, brp=args.brp)
    test_loader, _ = data_loader_all(num=args.ratio[2], bs=args.bs[2], file=args.te_file, lstm=args.lstm, brp=args.brp)

    best_model, best_info = train_val(net, epochs=args.epochs, train_loader=train_loader, valid_loader=valid_loader)
    test_acc_mse, lee_way = test(best_model, test_loader=test_loader, orig_data=_)

    print(f'best_info:{best_info}')
    print(f'test_acc_mse:{test_acc_mse}')

    if not args.brp:
        result_path = args.regress_results
        f = open(result_path, 'a+')
        csv_writer = csv.writer(f)
        csv_writer.writerow([str(args.comment), str(args.epochs), str(args.lstm), 'gpu', lee_way, time_, args.ratio,
                             args.mtl_ratio, test_acc_mse, best_info[0], best_info[1], best_info[2], best_info[3]])

    else:
        result_path = args.brp_results
        f = open(result_path, 'a+')
        csv_writer = csv.writer(f)
        csv_writer.writerow([str(args.brp), str(args.comment), time_, args.ratio, args.mtl_ratio,
                                 test_acc_mse, best_info[0], best_info[1], best_info[2], best_info[3]])


if __name__ == '__main__':
    main()
