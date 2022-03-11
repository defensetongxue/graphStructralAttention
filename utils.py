import numpy as np
import scipy.sparse as sp
import torch
import json as js
import pandas as pd 
def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    if dataset=='cora':
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        features  = normalize_features(features)
        labels = torch.LongTensor(np.where(labels)[1])
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    else: # dataset is chamelon
        file_prefix='./data/chameleon/chameleon_'
        feature_file=open(file_prefix+'features.json','r')
        data=js.load(feature_file)
        node_number=2277
        feature_length=3131 # featherlength is the length of noun in the data set
        features=np.zeros(shape=(node_number+1,feature_length+1))
        for i in range(node_number):
            for j in (data[str(i)]):
                features[i][j]=1
        file_label=open(file_prefix+'target.csv','r')
        labels=pd.read_csv(file_label)
        labels=np.array(labels.target)
        file_label=open(file_prefix+'edges.csv','r')
        edges=pd.read_csv(file_label)
        adj=np.eye(node_number+1)
        for i,j in zip(edges.id1,edges.id2):
            adj[i][j]=adj[j][i]=1
        adj=sp.csr_matrix(np.array(adj,dtype=np.float32),dtype=np.float32)
        features=sp.csr_matrix(features,dtype=np.float32)
        features  = normalize_features(features)
        idx_train = range(int(node_number*0.2))
        idx_val = range(int(node_number*0.2), int(node_number*0.4))
        idx_test = range(int(node_number*0.4), node_number)
    
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print("successfully loading data")
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

