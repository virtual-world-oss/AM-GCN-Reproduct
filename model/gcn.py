import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from torchinfo import summary

class GCNConv(nn.Module):
    def __init__(self,input_size,out_size,bias=True):
        super(GCNConv, self).__init__()
        # self.W = nn.Linear(input_size,out_size,bias=False)
        self.input_size = input_size
        self.out_size = out_size
        self.W = Parameter(torch.FloatTensor(input_size,out_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
        else:
            self.bias = None
        self.init_parameter()

    def init_parameter(self):
        border_val = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-border_val,border_val)
        if self.bias is not None:
            self.bias.data.uniform_(-border_val,border_val)

    def forward(self, A_, Z):
        x = torch.mm(Z,self.W)
        x = torch.mm(A_,x)
        # x = self.W(x)
        if self.bias is not None:
            x += self.bias
        return x

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes,dropout):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNConv(input_size,hidden_size)
        self.gcn_layer2 = GCNConv(hidden_size,num_classes)
        self.dropout = dropout

    def forward(self,A_,X):
        # print(X.shape)
        z = F.relu(self.gcn_layer1(A_,X))
        # print(z.shape)
        z = F.dropout(z,self.dropout,training=self.training)
        z = self.gcn_layer2(A_,z)
        out = F.log_softmax(z,dim=1)
        return out

if __name__ == '__main__':
    net = GCN(34,5,2)
    # print(net)
    features = np.eye(34,dtype='float')
    adj = np.ones((34,34),dtype='float')
    # print(features.shape)
    # print(adj.shape)
    features = torch.tensor(features,dtype=torch.float32)
    adj = torch.tensor(adj,dtype=torch.float32)
    out = net(adj,features)
    # print(out)
    # summary(net)
