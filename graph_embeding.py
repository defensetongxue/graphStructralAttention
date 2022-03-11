import numpy as np
import networkx as nx
import os
import pickle as pkl
from progress.bar import Bar
import torch
def dump_data(file_name,data):
    f=open('./Intermedium/'+file_name,'wb')
    pkl.dump(data,f)
    f.close()
def load_pkl_data(file_name):
    f=open('./interdata/'+file_name,'rb')
    data=pkl.load(f)
    f.close()
    return data


def cal_distance_matrix(adj,algorithm='dijstra',load_from_exisited=False,dataset="citeceer"):
    '''
    given a adjcent matrix and calculate the distance of each pair of node in a graph
    algorithm is now only for dijstra, for other algorithm is too slow, but we can try some parallel algorithm further
    return the distance matrix
    '''
    if load_from_exisited:
        assert os.path.isfile('Intermedium/distanceMatrix_{}.pkl'.format(dataset)),"distance matrix doesn't exisit"
        print("load distanceMatrix from existed file")    
        distance=load_pkl_data('distanceMatrix_{}.pkl'.format(dataset))
        return distance
    
    node_number=adj.shape[0]
    distance=adj.clone()
    edge=np.where(adj>0)
    if algorithm=="dijstra":
        print("calculate distance matrix which will cost a really long time")
        bar=Bar('processing',max=node_number)
        G=nx.DiGraph()
        for i in range(edge[0].shape[0]):
            G.add_edge(edge[0][i],edge[1][i],weight=1)
        for i in range(node_number):
            for j in range(node_number):
                try:
                    rs = nx.astar_path_length ( G,i, j )
                except nx.NetworkXNoPath:
                    rs=0
                distance[i][j]=rs
            bar.next()
        bar.finish()
    else:
        assert False,'''floyd is too slow'''
    dump_data('distanceMatrix_{}.pkl'.format(dataset),distance)
    print("finshed calculate distance matrix, begin to calculate ri_index and ri_all")
    return distance
        
def graph_embeding(adj,embeding_mode='rwr',c=0.5):
    node_number=adj.shape[0]
    if embeding_mode=='rwr':
        w=adj.clone()
        w=w/w.sum(0)
        E=torch.eye(node_number)
        graph_feature=((1-c)*((E-c*w).inverse())).T
        return graph_feature
    else:
        pass

