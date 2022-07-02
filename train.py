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
from sklearn.metrics import f1_score
import datetime
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



epochs = config['epoch']
lr = config['lr']
device = config['device']
dataset_name = config['dataset']
hiddensize1 = config['hiddensize1']
hiddensize2 = config['hiddensize2']
dropout = config['dropout']
weight_decay = config['weight_decay']
gama = config['gama']
beta = config['beta']
label_rate = config['label_rate']

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, idx_train, idx_val, idx_test, AF_ = load_data(dataset_name)
adj = adj.to(device)
features = features.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)
# train_mask = train_mask.to(device)
# val_mask = val_mask.to(device)
# test_mask = test_mask.to(device)

net = AMGCN(input_size=features.shape[1], hid_size1=hiddensize1, hid_size2=hiddensize2, num_classes=labels.shape[1], dropout=dropout)
net = net.to(device)

optimer = optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)

def cal_acc(out,label):
    out_label = out.max(1)[1].type_as(label)
    count_acc = out_label.eq(label).double()
    count_acc = count_acc.sum()
    return count_acc / len(label)

def cal_Lt(out,labels):
    # print(out)
    # print(labels)
    # return torch.FloatTensor()
    loss = 0.0
    for i in range(labels.shape[0]):
        loss += -(out[i][labels[i]])
    return loss / labels.shape[0]

def cal_Lc(Z_C_T, Z_C_F):
    Z_C_T = Z_C_T - torch.mean(Z_C_T,dim=0,keepdim=True)
    Z_C_F = Z_C_F - torch.mean(Z_C_F,dim=0,keepdim=True)
    ZCTnor = F.normalize(Z_C_T,p=2,dim=1)
    ZCFnor = F.normalize(Z_C_F,p=2,dim=1)
    ST = torch.mm(ZCTnor,ZCTnor.t())
    SF = torch.mm(ZCFnor,ZCFnor.t())
    Lc = torch.mean((ST - SF) ** 2)
    return Lc

def HSIC(mat1, mat2, n):
    R = torch.eye(n).to(device) - torch.ones(n, n).to(device) / n
    K1 = torch.mm(mat1,mat1.t())
    RK1 = torch.mm(R,K1)
    K2 = torch.mm(mat2,mat2.t())
    RK2 = torch.mm(R,K2)
    hsic = torch.trace(torch.mm(RK1,RK2))
    return hsic

def cal_Ld(Z_T,Z_C_T,Z_F,Z_C_F):
    Ld = (HSIC(Z_T, Z_C_T, Z_T.shape[0]) + HSIC(Z_F, Z_C_F, Z_F.shape[0])) / 2
    return Ld

def train(epoch):
    net.train()
    optimer.zero_grad()
    out, Z_C_F, Z_C_T, Z_T, Z_F = net(AF_, adj,features)

    if gama == 0:
        lc = None
    else:
        lc = cal_Lc(Z_C_T, Z_C_F)
    lt = F.nll_loss(out[idx_train], y_train)
    if beta == 0:
        ld = None
    else:
        ld = cal_Ld(Z_T,Z_C_T,Z_F,Z_C_F)
    loss_train = lt
    if lc is not None:
        loss_train = loss_train + gama*lc
    if ld is not None:
        loss_train = loss_train + beta*ld
    # loss_train = lt + gama*lc + beta*ld
    # loss_train = lt
    # loss_train = lt+ beta*ld
    # loss_train = cal_Lt(out[idx_train], y_train)
    # print(loss_train)
    # print(my_loss_train)
    acc_train = cal_acc(out[idx_train], y_train)
    loss_train.backward()
    optimer.step()

    acc_test, f1_test = mytest()
    # # loss_val = F.nll_loss(out[idx_val],y_val)
    # lt = F.nll_loss(out[idx_val],y_val)
    # # lc = cal_Lc(Z_C_T, Z_C_F)
    # # ld = cal_Ld(Z_T, Z_C_T, Z_F, Z_C_F)
    # loss_val = lt + gama * lc + beta * ld
    # acc_val = cal_acc(out[idx_val],y_val)

    print("epoch:{}\ttrain_loss:{}\ttrain_acc:{}\ttest_acc:{}\ttest_f1:{}".format(epoch+1,loss_train.item(),acc_train.item(),acc_test.item(),f1_test.item()))
    return acc_test, f1_test, acc_train, loss_train

def mytest():
    net.eval()
    out, Z_C_F, Z_C_T, Z_T, Z_F = net(AF_, adj,features)

    # # loss_test = F.nll_loss(out[idx_test],y_test)
    # lt = F.nll_loss(out[idx_test],y_test)
    # # lc = cal_Lc(Z_C_T, Z_C_F)
    # ld = cal_Ld(Z_T, Z_C_T, Z_F, Z_C_F)
    # # loss_test = lt + gama * lc + beta * ld
    # # loss_test = lt
    # loss_test = lt + beta * ld
    acc_test = cal_acc(out[idx_test],y_test)

    label_pred = []
    for index in idx_test:
        label_pred.append(torch.argmax(out[index]).item())
    label_true = y_test.tolist()
    # print(len(label_true))
    # print(label_pred.size())
    macro_f1 = f1_score(label_true,label_pred,average='macro')

    return acc_test,macro_f1
    # print("测试结果：loss:{},acc:{}".format(loss_test.item(),acc_test.item()))



# train(0)
# resfilename = 'dataset_{}_labelrate_{}.txt'.format(dataset_name,label_rate)
# resfilename = 'dataset_{}_labelrate_{}_wo.txt'.format(dataset_name,label_rate)
resfilename = 'dataset_{}_labelrate_{}_full.txt'.format(dataset_name,label_rate)
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
