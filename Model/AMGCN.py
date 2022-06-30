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

class gcn(nn.Module):
    def __init__(self,input_size,hid_size1,hid_size2,dropout):
        super(gcn, self).__init__()
        self.input_size = input_size
        self.hid_size1 = hid_size1
        self.hid_size2 = hid_size2
        self.dropout = dropout

        self.gcn_layer1 = GCNConv(input_size=input_size,out_size=hid_size1)
        self.gcn_layer2 = GCNConv(input_size=hid_size1,out_size=hid_size2)

    def forward(self,adj,features):
        z = F.relu(self.gcn_layer1(adj,features))
        z = F.dropout(z,self.dropout,training=self.training)
        z = self.gcn_layer2(adj,z)
        return z

# class common_gcn(nn.Module):
#     def __init__(self,input_size,hid_size1,hid_size2,dropout):
#         super(common_gcn, self).__init__()
#         self.input_size = input_size
#         self.hid_size1 = hid_size1
#         self.hid_size2 = hid_size2
#         self.dropout = dropout
#
#         self.gcn_layer1 = GCNConv(input_size=input_size, out_size=hid_size1)
#         self.gcn_layer2 = GCNConv(input_size=hid_size1, out_size=hid_size2)
#
#     def forward(self,adj,features):
#         z = F.relu(self.gcn_layer1(adj, features))
#         z = F.dropout(z, self.dropout, training=self.training)
#         z = self.gcn_layer2(adj, z)
#         return z

class Attention(nn.Module):
    def __init__(self,input_size, hid_size = 16):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hid_size = hid_size

        self.W_T = Parameter(torch.FloatTensor(hid_size, input_size))
        self.b_T = Parameter(torch.FloatTensor(hid_size,1))

        self.W_C = Parameter(torch.FloatTensor(hid_size, input_size))
        self.b_C = Parameter(torch.FloatTensor(hid_size, 1))

        self.W_F = Parameter(torch.FloatTensor(hid_size, input_size))
        self.b_F = Parameter(torch.FloatTensor(hid_size, 1))

        self.q = Parameter(torch.FloatTensor(hid_size,1))

        self.init_parameter()

    def init_parameter(self):
        border_val_t = 1. / math.sqrt(self.W_T.size(0))
        self.W_T.data.uniform_(-border_val_t,border_val_t)
        self.b_T.data.uniform_(-border_val_t,border_val_t)

        border_val_c = 1. / math.sqrt(self.W_C.size(0))
        self.W_C.data.uniform_(-border_val_c,border_val_c)
        self.b_C.data.uniform_(-border_val_c,border_val_c)

        border_val_f = 1. / math.sqrt(self.W_F.size(0))
        self.W_F.data.uniform_(-border_val_f,border_val_f)
        self.b_F.data.uniform_(-border_val_f,border_val_f)

        border_val_q = 1. / math.sqrt(self.q.size(0))
        self.q.data.uniform_(-border_val_q,border_val_q)

    def forward(self,Z_T,Z_C,Z_F):
        out_T = F.tanh(torch.mm(self.W_T,torch.t(Z_T)) + self.b_T)
        out_T = torch.mm(torch.t(self.q),out_T)
        out_T = F.softmax(out_T,dim=1)
        print(out_T.size())
        alpha_T = torch.diag_embed(out_T[0])

        out_C = F.tanh(torch.mm(self.W_C,torch.t(Z_C)) + self.b_C)
        out_C = torch.mm(torch.t(self.q), out_C)
        out_C = F.softmax(out_C,dim=1)
        print(out_C.size())
        alpha_C = torch.diag_embed(out_C[0])

        out_F = F.tanh(torch.mm(self.W_F,torch.t(Z_F)) + self.b_F)
        out_F = torch.mm(torch.t(self.q),out_F)
        out_F = F.softmax(out_F,dim=1)
        print(out_F.size())
        alpha_F = torch.diag_embed(out_F[0])

        print(alpha_F.size())
        print(alpha_T.size())
        print(alpha_C.size())

        out = torch.mm(alpha_T,Z_T) + torch.mm(alpha_C,Z_C) + torch.mm(alpha_F, Z_F)
        return out

# class classification_net


class AMGCN(nn.Module):
    def __init__(self,input_size,hid_size1,hid_size2,num_classes,dropout,att_hid_size = 16):
        super(AMGCN, self).__init__()
        self.input_size = input_size
        self.hid_size1 = hid_size1
        self.hid_size2 = hid_size2
        self.num_classes = num_classes
        self.dropout = dropout
        self.att_hid_size = att_hid_size

        self.special_gcn1 = gcn(input_size,hid_size1,hid_size2,dropout)
        self.special_gcn2 = gcn(input_size,hid_size1,hid_size2,dropout)
        self.common_gcn = gcn(input_size,hid_size1,hid_size2,dropout)

        self.att = Attention(hid_size2,att_hid_size)

        self.lin = nn.Linear(hid_size2,num_classes)

    def forward(self,AF_,AT_,features):
        Z_T = self.special_gcn1(AT_,features)
        Z_F = self.special_gcn2(AF_,features)
        Z_C_T = self.common_gcn(AT_,features)
        Z_C_F = self.common_gcn(AF_,features)
        Z_C = (Z_C_F + Z_C_T) / 2

        Z = self.att(Z_T, Z_C, Z_F)
        out = self.lin(Z)

        return F.log_softmax(out), Z_C_F, Z_C_T, Z_T, Z_F

# class GCN(nn.Module):
#     def __init__(self,input_size,hidden_size,num_classes,dropout):
#         super(GCN, self).__init__()
#         self.gcn_layer1 = GCNConv(input_size,hidden_size)
#         self.gcn_layer2 = GCNConv(hidden_size,num_classes)
#         self.dropout = dropout
#
#     def forward(self,A_,X):
#         # print(X.shape)
#         z = F.relu(self.gcn_layer1(A_,X))
#         # print(z.shape)
#         z = F.dropout(z,self.dropout,training=self.training)
#         z = self.gcn_layer2(A_,z)
#         out = F.log_softmax(z,dim=1)
#         return out

if __name__ == '__main__':
    # net = GCN(34,5,2)
    # # print(net)
    # features = np.eye(34,dtype='float')
    # adj = np.ones((34,34),dtype='float')
    # # print(features.shape)
    # # print(adj.shape)
    # features = torch.tensor(features,dtype=torch.float32)
    # adj = torch.tensor(adj,dtype=torch.float32)
    # out = net(adj,features)
    # print(out)
    # summary(net)
    # a = np.array([[1,2,3],[4,5,6]])
    # a = torch.FloatTensor(a)

    # print(a)
    # b = np.array([[1],[2]])
    # b = torch.tensor(b)
    # print(b)
    # c = a+b
    # print(c)
    # b = torch.t(b)
    # print(torch.t(c))
    # print(b[0])
    # b = torch.diag_embed(b[0])
    # print(b)
    # print(b.size())
    # net = nn.Linear()
    # b = net(a)
    # print(b.size())

    pass