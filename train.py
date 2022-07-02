from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataloader import load_data
from models import GAT, SpGAT
from config import config
from sklearn.metrics import f1_score
import time
import datetime

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

dataset_name = config['dataset']
label_rate = config['label_rate']
epochs = 50

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, idx_train, idx_val, idx_test = load_data(
    config['dataset'])

# Model and optimizer
if args.sparse:
    net = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden, 
                nclass=labels.shape[1],
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    net = GAT(nfeat=features.shape[1],
                nhid=args.hidden, 
                nclass=labels.shape[1],
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimer = optim.Adam(net.parameters(),
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    net.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def cal_acc(out,label):
    out_label = out.max(1)[1].type_as(label)
    count_acc = out_label.eq(label).double()
    count_acc = count_acc.sum()
    return count_acc / len(label)

def train(epoch):
    net.train()
    optimer.zero_grad()
    out = net(features,adj)
    # print(y_train)
    loss_train = F.nll_loss(out[idx_train],y_train)
    acc_train = cal_acc(out[idx_train],y_train)
    loss_train.backward()
    optimer.step()


    # loss_val = F.nll_loss(out[idx_val],y_val)
    # acc_val = cal_acc(out[idx_val],y_val)
    acc_test, f1_test = mytest()
    print("epoch:{}\ttrain_loss:{}\ttrain_acc:{}\ttest_acc:{}\ttest_f1:{}".format(epoch + 1, loss_train.item(),
                                                                                  acc_train.item(), acc_test.item(),
                                                                                  f1_test.item()))
    return acc_test, f1_test, acc_train, loss_train
def mytest():
    net.eval()
    out = net(features,adj)
    loss_test = F.nll_loss(out[idx_test],y_test)
    acc_test = cal_acc(out[idx_test],y_test)

    label_pred = []
    for index in idx_test:
        label_pred.append(torch.argmax(out[index]).item())
    label_true = y_test.tolist()
    # print(len(label_true))
    # print(label_pred.size())
    macro_f1 = f1_score(label_true, label_pred, average='macro')

    return acc_test, macro_f1


resfilename = 'dataset_{}_labelrate_{}.txt'.format(dataset_name,label_rate)
# print(resfilename)
if not os.path.exists(resfilename):
    f = open(resfilename,'w')
    f.close()


with open('log.txt','a') as logfile:
    nowtime = datetime.datetime.now()
    logfile.write(str(nowtime))
    logfile.write('\n')

    t = time.time()
    max_acc = 0.0
    max_f1 = 0.0
    optimal_epoch = 0
    for epoch in range(epochs):
        acc_test, f1_test, acc_train, loss_train = train(epoch)
        train_log = "epoch:{}\ttrain_loss:{}\ttrain_acc:{}\ttest_acc:{}\ttest_f1:{}\n".format(epoch+1,loss_train.item(),acc_train.item(),acc_test.item(),f1_test.item())
        logfile.write(train_log)
        if acc_test >= max_acc:
            max_acc = acc_test
            max_f1 = f1_test
            optimal_epoch = epoch+1
    print("训练并测试结束，共训练{}轮，总用时{}s".format(epochs,time.time()-t))
    print("最佳正确率为:{},对应的macro_f1为:{},对应的训练轮次为:{}".format(max_acc,max_f1,optimal_epoch))
    endlog = "训练并测试结束，共训练{}轮，总用时{}s\n".format(epochs,time.time()-t)
    logfile.write(endlog)
    endlog = "最佳正确率为:{},对应的macro_f1为:{},对应的训练轮次为:{}\n".format(max_acc,max_f1,optimal_epoch)
    logfile.write(endlog)
    logfile.write('\n\n\n')
    # print("开始在测试集上进行测试")
    # mytest()
with open(resfilename,'a') as resfile:
    nowtime = datetime.datetime.now()
    resfile.write(str(nowtime))
    resfile.write('\n')
    resfile.write('当前超参数与配置为：\n')
    resfile.write(str(config))
    resfile.write('\n本次训练测试结果为：\n')
    res_str = "最佳正确率为:{},对应的macro_f1为:{},对应的训练轮次为:{}\n".format(max_acc,max_f1,optimal_epoch)
    resfile.write(res_str)
    resfile.write('\n\n\n')