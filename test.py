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

# a = torch.randn(3,3)
# print(a)
# b = torch.randn(3,3)
# print(b)
# c = (a - b) ** 2
# print(c)

# print()
# if not os.path.exists('result.txt'):
#     f = open('result.txt','w')
#     f.close()
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

# resfilename = 'dataset_{}_labelrate_{}.txt'.format(dataset_name,label_rate)
# print(resfilename)
# if not os.path.exists(resfilename):
#     f = open(resfilename,'w')
#     f.close()

import datetime
nowtime = datetime.datetime.now()
print(nowtime)
