import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
# from dataloader import load_data

def to_tuple_sigle(matrix):
    if not sp.isspmatrix_coo(matrix):
        matrix = matrix.tocoo()
    coords = np.vstack((matrix.row,matrix.col)).transpose()
    values = matrix.data
    shape = matrix.shape
    return coords, values, shape

def sparse_to_tuple(sp_matrix):
    if isinstance(sp_matrix,list):
        for i in range(len(sp_matrix)):
            sp_matrix[i] = to_tuple_sigle(sp_matrix[i])
    else:
        sp_matrix = to_tuple_sigle(sp_matrix)
    return sp_matrix

def sparse_to_np(matrix):
    return matrix.toarray()

def np_to_tensor(matrix,dtype = torch.float32):
    return torch.tensor(matrix, dtype=dtype)

def get_A_(adj):
    # adj = adj + np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    degree = sp.coo_matrix(degree)
    adj = sp.coo_matrix(adj)
    return degree.dot(adj).dot(degree).toarray()

def normlize(matrix):
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum,-1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    matrix = r_mat_inv.dot(matrix)
    return matrix

if __name__ == '__main__':
    from dataloader import load_data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, idx_train, idx_val, idx_test = load_data('cora')
    # print(type(adj))
    # print(type(features))
    adj = sparse_to_np(adj)
    features = sparse_to_np(features)
    # print(adj.shape)
    # print(features.shape)
    # print(y_train)
    # print(type(y_val))
    # print(type(y_test))
    # print(type(train_mask))
    # print(type(val_mask))
    # print(type(test_mask))
    pass