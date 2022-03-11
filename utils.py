import imp
import numpy as np
import scipy.sparse as sp
import torch
import json as js
import pandas as pd 
from data_loader.cora_loader import cora_loader
from data_loader.citeseer_loader import citeceer_loader


def load_data(dataset="cora",train_test_val=[0.2,0.2,0.6]):
    if dataset=="cora":
        adj, features, labels, idx_train, idx_val, idx_test=cora_loader()
    elif dataset=='citeseer':
        adj, features, labels, idx_train, idx_val, idx_test=citeceer_loader(train_test_val)
    else:
        assert False,"don't have that dataset"
    return adj, features, labels, idx_train, idx_val, idx_test



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

