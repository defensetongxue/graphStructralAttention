from copyreg import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class mlp(nn.Moudule):
    def __init__(self,in_features,out_features,act_layer=nn.LeakyReLu):
        super(mlp,self).__init__()
        self.layer1=nn.Linear(in_features, out_features)
        self.act_layer=act_layer()
    def forward(self,x):
        x=self.layer1(x)
        x=act_layer(x)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        #torch.save(Wh,'out_features.pkl')
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #torch.save(attention,'att_score.pkl')
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class researchModel(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features1,in_features2, out_features1,out_features2, out_features ,dropout, alpha, concat=True):
        super(researchModel, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features1, out_features1)))
        self.W2 = nn.Parameter(torch.empty(size=(in_features2, out_features2)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=(2*out_features1, 1)))
        self.a2 = nn.Parameter(torch.empty(size=(2*out_features2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.mlp=mlp(out_features1+out_features2,out_features)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh1 = torch.mm(h[:,:in_features1], self.W1) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e1 = self._prepare_attentional_mechanism_input(Wh1)
        #torch.save(Wh,'out_features.pkl')
        zero_vec = -9e15*torch.ones_like(e1)
        attention1 = torch.where(adj > 0, e1, zero_vec)
        attention1 = F.softmax(attention1, dim=1)
        #torch.save(attention,'att_score.pkl')
        attention1 = F.dropout(attention1, self.dropout, training=self.training)
        h_prime1=torch.matmul(attention1, Wh1)

        Wh2 = torch.mm(h[:,in_features1:], self.W2) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e2 = self._prepare_graph_attentional_mechanism_input(Wh2)
        #torch.save(Wh,'out_features.pkl')
        zero_vec = -9e15*torch.ones_like(e2)
        attention2 = torch.where(adj > 0, e2, zero_vec)
        attention2 = F.softmax(attention2, dim=1)
        #torch.save(attention,'att_score.pkl')
        attention2 = F.dropout(attentio2n, self.dropout, training=self.training)
        h_prime2 = torch.matmul(attention2, Wh2)
        h_prime=torch.cat([h_prime1,h_primw2],dim=1)
        if self.concat:
            h_prime=F.elu(h_prime)
            return self.act_layer(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a1[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a1[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    def _prepare_graph_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a2[:self.out_features2, :])
        Wh2 = torch.matmul(Wh, self.a2[self.out_features2:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
