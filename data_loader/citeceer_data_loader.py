from turtle import distance
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd

def citeceer_load_data(train_val_test=[0.2,0.2,0.6]):
    """loading data from the data set """
    print('Loading citeceer dataset...')
    # get the data from dataset
    part_sum=train_val_test[0]+train_val_test[1]+train_val_test[2]
    assert part_sum==1,"sum of train,val,test should be one "
    data_content = pd.read_csv('../data/citeseer/citeseer.content',sep='\t',header=None)
    data_edge = pd.read_csv('../data/citeseer/citeseer.cites',sep='\t',header=None)

    data_idx=list(data_content.index)
    paper_id=list(data_content.iloc[:,0])
    data_map=dict(zip(paper_id,data_idx))
    bad_data_index=[]
    for i in range(data_edge.shape[0]):
        if (not data_edge.iloc[i][0] in data_map.keys()) or (not data_edge.iloc[i][1] in data_map.keys()) or  (data_edge.iloc[i][1] ==data_edge.iloc[i][0]): 
            bad_data_index.append(i)
    data_edge=data_edge.drop(bad_data_index,axis=0)
    data_edge=data_edge.applymap(data_map.get)
    labels=data_content.iloc[:,-1]
    labels=pd.get_dummies(labels)
    features= data_content.iloc[:,1:-1]
    node_number=data_content.shape[0]
    adj=np.eye(node_number)
    for i,j in zip(data_edge[0],data_edge[1]):
        adj[i][j]=adj[j][i]=1

    # build graph# idx_map is maping the index of city to the consecutive integer
    
    idx_test = range(int(node_number*train_val_test[0]))
    idx_train = range(int(node_number*train_val_test[0]),int(node_number*train_val_test[1]+node_number*train_val_test[0]))
    idx_val = range(int(node_number*train_val_test[1]+node_number*train_val_test[0]), node_number)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj = torch.FloatTensor(np.array(adj))
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])

    adj=normalize_adj(adj)
    features=normalize_features(features)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    print("successfully loading data from dataset citeceer")
    return adj, features,idx_train, idx_val, idx_test, labels

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx=sp.csr_matrix(mx)
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


