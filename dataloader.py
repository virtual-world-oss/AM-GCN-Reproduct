import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from utils import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from config import config

def get_per_line_idx(filename):
    idx = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            idx.append(int(line))
    return idx

def get_mask_matrix(idx, shape):
    mask = np.zeros(shape)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_train_node(label_rate,allx,ally):
    # print(ally)
    # print(allx.shape)
    allx = allx.toarray()
    # ally = ally.toarray()
    tmp_y = np.where(ally)[1]
    # print(tmp_y)
    classes = {}
    for i in range(ally.shape[1]):
        classes[i] = []
    # print(classes)
    for i in range(tmp_y.shape[0]):
        classes[tmp_y[i]].append(i)
    # classes[tmp_y[0]].append(0)
    # print(classes[3])
    # print(classes[4])
    train_idx = []
    # train_idx += classes[3][:label_rate]
    # train_idx += classes[4][:label_rate]
    for i in range(ally.shape[1]):
        train_idx += classes[i][:label_rate]
    # print(len(train_idx))
    # train_x = allx[train_idx,:]
    # train_y = ally[train_idx,:]
    # print(train_x.shape)
    # print(train_idx)
    # print(max(train_idx))
    # print(type(allx))
    # print(ally)
    # other_x = np.delete(allx, train_idx, axis=0)
    # other_y = np.delete(ally, train_idx, axis=0)
    # # print(other_x.shape)
    # # print(other_y.shape)
    # new_allx = np.vstack((train_x,other_x))
    # new_ally = np.vstack((train_y,other_y))
    # print(new_ally.shape)
    # print(new_allx.shape)
    # return train_x,train_y,new_allx,new_ally
    return train_idx

def load_data(dataset_name):
    backnames = ['x','y','tx','ty','allx','ally','graph']
    objects = []

    '''
    将数据集中的数据加载进来
    '''
    for i in range(len(backnames)):
        with open('data/ind.{}.{}'.format(dataset_name, backnames[i]), 'rb') as file:
            # print(file.name)
            if sys.version_info > (3, 0):
                objects.append(pkl.load(file, encoding='latin1'))
            else:
                objects.append(pkl.load(file))
    # print(type(objects[0]))
    '''
    加载结束
    '''

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # print(type(x))
    # print(type(y))


    # 将各个文件中的数据分开加载进来
    test_idx_reorder = get_per_line_idx("data/ind.{}.test.index".format(dataset_name))
    test_idx_range = np.sort(test_idx_reorder)
    # print(test_idx_sort)
    # print(ally)
    if dataset_name == 'citeseer':
        # 在我参考的代码中，解释到因为citeseer数据集有独立的节点，需要把独立的节点作为零向量插入到正确的位置，还是数据集处理的问题
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        # print(test_idx_range)
        # print(x.shape)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # print(tx_extended.shape)
        # print(tx.shape)
        tx_extended[test_idx_range-min(test_idx_range),:] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full),y.shape[1]))

        # print(ty.shape)
        ty_extended[test_idx_range - min(test_idx_range),:] = ty
        ty = ty_extended
        # print(ty_extended.shape)

    features = sp.vstack((allx,tx)).tolil()
    # print(len(test_idx_reorder))
    # print(features.shape)
    # print(features[
    # test_idx_reorder,:])
    # print(features[test_idx_range, :])
    features[test_idx_reorder,:] = features[test_idx_range,:]
    # print(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # print(adj.shape)

    labels = np.vstack((ally,ty))
    labels[test_idx_reorder,:] = labels[test_idx_range,:]

    idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y),len(y)+500)

    # train_num = config['label_rate'] * ally.shape[1]
    # idx_train = list(range(train_num))
    # idx_val = list(range(train_num, train_num + 500))

    train_idx_reorder = get_train_node(config['label_rate'], allx, ally)
    # print(train_idx_reorder)
    all_index = list(range(len(ally)))
    # print(all_index)
    idx_not_train = [i for i in all_index if i not in train_idx_reorder]
    val_idx_reorder = idx_not_train[:500]
    # print(val_idx_reorder)
    train_idx_range = np.sort(train_idx_reorder)
    val_idx_range = np.sort(val_idx_reorder)
    # print(train_idx_range)
    # print(val_idx_range)
    # features[train_idx_reorder, :] = features[train_idx_range, :]
    # features[val_idx_reorder, :] = features[val_idx_range, :]
    # labels[train_idx_reorder, :] = labels[train_idx_range, :]
    # labels[val_idx_reorder, :] = labels[val_idx_range, :]
    idx_train = train_idx_range.tolist()
    idx_val = val_idx_range.tolist()
    # train_num = config['label_rate'] * ally.shape[1]
    # idx_train = list(range(train_num))
    # idx_val = list(range(train_num, train_num + 500))

    # print(idx_train)
    # print(idx_val)
    # print(len(idx_val))
    # print(len(idx_train))

    # print(features.shape)
    # print(labels.shape)

    # train_mask = get_mask_matrix(idx_train, labels.shape[0])
    # val_mask = get_mask_matrix(idx_val,labels.shape[0])
    # test_mask = get_mask_matrix(idx_test,labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[idx_train, :] = labels[idx_train, :]
    y_val[idx_val, :] = labels[idx_val, :]
    y_test[idx_test, :] = labels[idx_test, :]

    # print(y_train.shape)

    adj = adj + sp.eye(adj.shape[0])

    features = normlize(features)
    adj = normlize(adj)

    adj = get_A_(sparse_to_np(adj))
    features = sparse_to_np(features)


    adj = np_to_tensor(adj)
    features = np_to_tensor(features)
    y_train = np_to_tensor(np.where(y_train)[1],dtype=torch.long)
    y_val = np_to_tensor(np.where(y_val)[1],dtype=torch.long)
    y_test = np_to_tensor(np.where(y_test)[1],dtype=torch.long)
    # train_mask = np_to_tensor(train_mask,dtype=torch.long)
    # val_mask = np_to_tensor(val_mask,dtype=torch.long)
    # test_mask = np_to_tensor(test_mask,dtype=torch.long)
    train_mask = []
    val_mask = []
    test_mask = []

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, idx_train, idx_val, idx_test

if __name__ == '__main__':
    # load_data('citeseer')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels, idx_train, idx_val, idx_test = load_data('cora')
    # # print(y_train)
    # x = np.where(y_train)
    # x = features[idx_train]
    # print(x.shape)
    # print(labels.shape)
    # print(len(idx_train))
    # # print(train_mask)
    # print(adj.shape)
    # print(labels.shape)
