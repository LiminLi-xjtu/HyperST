import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from layers import GraphConvolution, HGNN_conv,UniGATConv
from sklearn.cluster import KMeans
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        for layer in self.project:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class Attention1(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention1, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
        

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class HyperST(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels, heads=1, dropout=0.):
        super(HyperST, self).__init__()
        self.conv1_1 = HGNN_conv(in_channels, hidden_channels)
        self.conv1_2 = UniGATConv(args, in_channels * heads, hidden_channels, heads, dropout)
        self.conv1_3 = GraphConvolution(in_channels, hidden_channels)
        self.conv1_4 = GraphConvolution(in_channels, hidden_channels)
    
        self.conv2_1 = HGNN_conv(hidden_channels, out_channels)
        self.conv2_2 = UniGATConv(args, hidden_channels * heads, out_channels, heads, dropout)
        self.conv2_3 = GraphConvolution(hidden_channels, out_channels)
        self.conv2_4 = GraphConvolution(hidden_channels, out_channels)

        self.decoder1 = GraphConvolution(out_channels,in_channels)

        self.att = Attention(hidden_channels)
        self.att_1 = Attention1(out_channels)
        self.args=args
        
    def forward(self, X_gene, H_gene,vertex_spa, edges_spa,G_gene,G_spatial):
        #第一层
        H1 = self.conv1_1(X_gene, H_gene)
        H2 = self.conv1_2(X_gene, vertex_spa, edges_spa)
        h1 = F.relu(self.conv1_3(X_gene,G_gene))
        s1 = F.relu(self.conv1_4(X_gene,G_spatial))
        
        H3=torch.stack([H1,H2],dim=1)
        H3,att1=self.att(H3)

        # 第二层
        out1 = self.conv2_1(H3, H_gene)
        out2 = self.conv2_2(H1, vertex_spa, edges_spa)
        h2 = self.conv2_3(h1,G_gene)
        s2 = self.conv2_4(s1,G_spatial)

        out_atten=torch.stack([out2,out1],dim=1)
        out_atten,att=self.att_1(out_atten)

        de_X_gene=self.decoder1(h2,G_gene)

        return h2, s2, out_atten ,de_X_gene,out1,out2
 

class HyperST_sparse(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels, heads=1, dropout=0.):
        super(HyperST_sparse, self).__init__()
        self.conv1_1 = HGNN_conv(in_channels, hidden_channels)
        self.conv1_2 = UniGATConv(args, in_channels * heads, hidden_channels, heads, dropout)

        self.conv2_1 = HGNN_conv(hidden_channels, out_channels)
        self.conv2_2 = UniGATConv(args, hidden_channels * heads, out_channels, heads, dropout)

        self.att = Attention1(out_channels)
        self.att_1=Attention1(hidden_channels,hidden_size=16)
    
        self.args=args
        
    def forward(self, X_gene, H_gene,vertex_spa, edges_spa):
        #第一层
        H1 = self.conv1_1(X_gene, H_gene)
        H2 = self.conv1_2(X_gene, vertex_spa, edges_spa)

        H3=torch.stack([H1,H2],dim=1)
        H3,att1=self.att(H3)

        # 第二层
        out1 = self.conv2_1(H3, H_gene)
        out2 = self.conv2_2(H1, vertex_spa, edges_spa)

        out_atten=torch.stack([out2,out1],dim=1)
        out_atten,att=self.att_1(out_atten)

        return out1, out2, out_atten 