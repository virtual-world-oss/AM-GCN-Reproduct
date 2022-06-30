import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from torchinfo import summary
from dataloader import load_data
# from model.gcn import GCN
import time
from Model.AMGCN import AMGCN
from config import config
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



epochs = config['epoch']
lr = config['lr']
device = config['device']
dataset_name = config['dataset']
hiddensize = config['hiddensize']
dropout = config['dropout']
weight_decay = config['weight_decay']

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, idx_train, idx_val, idx_test, AF_ = load_data(dataset_name)
adj = adj.to(device)
features = features.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)
# train_mask = train_mask.to(device)
# val_mask = val_mask.to(device)
# test_mask = test_mask.to(device)

net = GCN(features.shape[1],hiddensize,labels.shape[1],dropout)
net = net.to(device)

optimer = optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)

def cal_acc(out,label):
    out_label = out.max(1)[1].type_as(label)
    count_acc = out_label.eq(label).double()
    count_acc = count_acc.sum()
    return count_acc / len(label)

def train(epoch):
    net.train()
    optimer.zero_grad()
    out = net(adj,features)
    loss_train = F.nll_loss(out[idx_train],y_train)
    acc_train = cal_acc(out[idx_train],y_train)
    loss_train.backward()
    optimer.step()


    loss_val = F.nll_loss(out[idx_val],y_val)
    acc_val = cal_acc(out[idx_val],y_val)

    print("epoch:{}\ttrain_loss:{}\ttrain_acc:{}\tval_loss:{}\tval_acc:{}".format(epoch+1,loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item()))

def mytest():
    net.eval()
    out = net(adj,features)
    loss_test = F.nll_loss(out[idx_test],y_test)
    acc_test = cal_acc(out[idx_test],y_test)
    print("测试结果：loss:{},acc:{}".format(loss_test.item(),acc_test.item()))


t = time.time()
for epoch in range(epochs):
    train(epoch)
print("训练结束，共训练{}轮，总用时{}s".format(epochs,time.time()-t))
print("开始在测试集上进行测试")
mytest()
