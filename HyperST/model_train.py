import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch_sparse
from torch.nn.parameter import Parameter
from tqdm import tqdm
from model import *
from utils import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def get_q(self, z):
    q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.beta) + 1e-8)
    q = q**(self.beta+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q

def target_distribution(q):
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p

def iterative_loss(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
    loss = kld(p, q)
    return loss

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


# def cosine_similarity(emb):
#     mat = torch.matmul(emb, emb.T)
#     norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
#     mat = torch.div(mat, torch.matmul(norm, norm.T))
#     if torch.any(torch.isnan(mat)):
#         mat = _nan2zero(mat)
#     mat = mat - torch.diag_embed(torch.diag(mat))
#     return mat


# InfoNCE Loss Function
from torch.nn.functional import cosine_similarity
def info_nce_loss(x_H, x_A, tau=0.2):
    """
    Compute InfoNCE loss for collaborative contrastive learning.

    Parameters:
        x_H (torch.Tensor): Node representations from HGCN.
        x_A (torch.Tensor): Node representations from GCN.
        tau (float): Temperature parameter.

    Returns:
        torch.Tensor: InfoNCE loss.
    """
    batch_size = x_H.size(0)
    sim_matrix = torch.sigmoid(torch.matmul(x_H, x_A.T))

    # Positive pairs: diagonal elements (same nodes)
    positive_pairs = torch.diag(sim_matrix)

    # Negative pairs: off-diagonal elements
    negative_pairs = sim_matrix - torch.eye(batch_size, device=x_H.device) * sim_matrix
    negative_pairs = torch.clamp(negative_pairs, 1e-5, 1e6)

    # Numerator and denominator for InfoNCE
    numerator = torch.exp(positive_pairs / tau)
    denominator = torch.sum(torch.exp(negative_pairs/ tau), dim=1)

    # Compute InfoNCE loss
    loss = -torch.log(numerator / denominator)
    return loss.mean()


def normalize_adj(adj):

    row_sum = adj.sum(1).clamp(min=1e-12)  
    deg_inv_sqrt = torch.pow(row_sum, -0.5)
    
    # normalization
    deg_inv_sqrt_col = deg_inv_sqrt.view(-1, 1)  
    deg_inv_sqrt_row = deg_inv_sqrt.view(1, -1)  
    norm_adj = deg_inv_sqrt_col * adj * deg_inv_sqrt_row
    
    return norm_adj

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn
    
class HyperSTModel_Train:
    def __init__(self, adata, incidence_gene,incidence_spatial,params,init="mclust",datatype='10X',seed = 818):
        self.params = params
        self.device = params.device
        self.n_clusters=params.n_domains
        self.max_epochs=params.epochs
        self.embedding_size=params.hidden_dim2
        self.init=init
        self.seed=seed
        self.beta=0.5
        self.datatype=datatype

        # self.X = torch.FloatTensor(adata.X.A).to(self.device)
        self.X = torch.FloatTensor(adata.obsm['X_pca']).to(self.device)

        (row_gene, col_gene), value_gene = torch_sparse.from_scipy(incidence_gene)
        H_gene=incidence_gene.toarray().astype(np.float32)
        G_gene=generate_G_from_H(H_gene)
        # self.row_gene=row_gene.to(self.device)
        # self.col_gene=col_gene.to(self.device)
        self.G_gene=torch.FloatTensor(G_gene).to(self.device)
        self.H_gene=torch.FloatTensor(H_gene).to(self.device)

        if self.datatype not in ['Stereo','Slide-seq']:
            self.G_gene111=torch.matmul(self.H_gene,self.H_gene.T)

        (row_spa, col_spa), value_spa = torch_sparse.from_scipy(incidence_spatial)
        H_spatial=incidence_spatial.toarray()
        self.H_spatial=torch.FloatTensor(H_spatial).to(self.device)

        if self.datatype not in ['Stereo','Slide-seq']:
            self.G_spatial=torch.matmul(self.H_spatial,self.H_spatial.T)

        self.row_spa=row_spa.to(self.device)
        self.col_spa=col_spa.to(self.device)
        # self.G_spatial=torch.FloatTensor(G_spatial).to(self.device)

        if self.datatype in ['Stereo','Slide-seq']:
            self.model=HyperST_sparse(args=params,in_channels=params.latent_dim,out_channels=params.hidden_dim2, hidden_channels=params.hidden_dim1).to(self.device)
        else:
            self.model=HyperST(args=params,in_channels=params.latent_dim,out_channels=params.hidden_dim2, hidden_channels=params.hidden_dim1).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.params.lr, weight_decay=self.params.weight_decay)

    def training_model(self):
        with torch.no_grad():
            _, _,features,_,_,_ = self.model(self.X,self.G_gene,self.row_spa,self.col_spa,self.G_gene111,self.G_spatial)
     
        #----------------------------------------------------------------           
        if self.init=="kmeans":
            print("Initializing cluster centers with kmeans")
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())
            
        elif self.init=="mclust":
            print("Initializing cluster centers with mclust")
            data = features.detach().cpu().numpy()
            y_pred = mclust(data, num_cluster = self.n_clusters, random_seed = self.seed)
            y_pred = y_pred.astype(int)

        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.embedding_size))

        features = pd.DataFrame(features.detach().cpu().numpy()).reset_index(drop = True)
        Group = pd.Series(y_pred, index=np.arange(0,features.shape[0]), name="Group")
        Mergefeature = pd.concat([features,Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())       
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        
        #---------------------------------------------------------------- 
        with tqdm(total=self.max_epochs) as t:
            for epoch in range(self.max_epochs):            
                t.set_description('Epoch')
                self.model.train()
                
                if epoch%self.params.update_interval == 0:
                    _, _,embed,_,_,_ = self.model(self.X,self.G_gene,self.row_spa,self.col_spa,self.G_gene111,self.G_spatial)
                    Q = get_q(self,embed)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    t.update(self.params.update_interval)
                    
                z_gene, z_spa,embed,X_dec,_,_ = self.model(self.X,self.G_gene,self.row_spa,self.col_spa,self.G_gene111,self.G_spatial)
                q = get_q(self,embed)
                p = target_distribution(Q.detach())
            
                KL_loss=iterative_loss(p, q)
                info_loss1=info_nce_loss(z_spa,embed)
                info_loss2=info_nce_loss(z_gene,embed)
                reco_loss1=reconstruction_loss(X_dec,self.X)
                loss=1.0*KL_loss+0.1*info_loss1+0.1*info_loss2+0.1*reco_loss1
                

                self.optimizer.zero_grad()              
                loss.backward()

                self.optimizer.step()
    
                t.set_postfix(loss = loss.data.cpu().numpy())
                
                #Check stop criterion
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / self.X.shape[0]
                y_pred_last = y_pred
                if epoch>0 and (epoch-1)%self.params.update_interval == 0 and delta_label < self.params.tol:
                    print('delta_label ', delta_label, '< tol ', self.params.tol)
                    print("Reach tolerance threshold. Stopping training.")
                    print("Total epoch:", epoch)
                    break

    
    def predict(self):
        self.model.eval()
        _, s2,z,_,_,_= self.model(self.X,self.G_gene,self.row_spa,self.col_spa,self.G_gene111,self.G_spatial)
        q=get_q(self,z)
        latent_z = z.data.cpu().numpy()
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        prob=q.data.cpu().numpy()
        s2_a=s2.data.cpu().numpy()
        return y_pred,latent_z,prob
      

class HyperSTModel_Train_sparse:
    def __init__(self, adata, incidence_gene,incidence_spatial,params,init="mclust",seed = 818):
        self.params = params
        self.device = params.device
        self.n_clusters=params.n_domains
        self.max_epochs=params.epochs
        self.embedding_size=params.latent_dim
        self.init=init
        self.seed=seed
        self.beta=0.5

        # self.X = torch.FloatTensor(adata.X.A).to(self.device)
        self.X = torch.FloatTensor(adata.obsm['X_pca']).to(self.device)

        # (row_gene, col_gene), value_gene = torch_sparse.from_scipy(incidence_gene)
        H_gene=incidence_gene.toarray().astype(np.float32)
        G_gene=generate_G_from_H(H_gene)
        # self.row_gene=row_gene.to(self.device)
        # self.col_gene=col_gene.to(self.device)
        self.G_gene=torch.FloatTensor(G_gene).to(self.device)

        (row_spa, col_spa), value_spa = torch_sparse.from_scipy(incidence_spatial)
        # H_spatial=incidence_spatial.toarray()
        # self.H_spatial=torch.FloatTensor(H_spatial).to(self.device)
        self.row_spa=row_spa.to(self.device)
        self.col_spa=col_spa.to(self.device)

        self.model=HyperST_sparse(args=params,in_channels=params.latent_dim,out_channels=params.latent_dim, hidden_channels=params.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.params.lr, weight_decay=self.params.weight_decay)

    def training_model(self):
        with torch.no_grad():
            _, _,features = self.model(self.X,self.G_gene,self.row_spa,self.col_spa)
     
        #----------------------------------------------------------------           
        if self.init=="kmeans":
            print("Initializing cluster centers with kmeans")
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())
            
        elif self.init=="mclust":
            print("Initializing cluster centers with mclust")
            data = features.detach().cpu().numpy()
            y_pred = mclust(data, num_cluster = self.n_clusters, random_seed = self.seed)
            y_pred = y_pred.astype(int)

        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.embedding_size))

        features = pd.DataFrame(features.detach().cpu().numpy()).reset_index(drop = True)
        Group = pd.Series(y_pred, index=np.arange(0,features.shape[0]), name="Group")
        Mergefeature = pd.concat([features,Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())       
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        
        #---------------------------------------------------------------- 
        with tqdm(total=self.max_epochs) as t:
            for epoch in range(self.max_epochs):            
                t.set_description('Epoch')
                self.model.train()
                
                if epoch%self.params.update_interval == 0:
                    _, _,embed = self.model(self.X,self.G_gene,self.row_spa,self.col_spa)
                    Q = get_q(self,embed)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    t.update(self.params.update_interval)
                    
                z_gene, z_spa,embed= self.model(self.X,self.G_gene,self.row_spa,self.col_spa)
                q = get_q(self,embed)
                p = target_distribution(Q.detach())
                
                KL_loss=iterative_loss(p, q)
                loss=KL_loss

                self.optimizer.zero_grad()              
                loss.backward()
                self.optimizer.step()
    
                t.set_postfix(loss = loss.data.cpu().numpy())
                
                #Check stop criterion
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / self.X.shape[0]
                y_pred_last = y_pred
                if epoch>0 and (epoch-1)%self.params.update_interval == 0 and delta_label < self.params.tol:
                    print('delta_label ', delta_label, '< tol ', self.params.tol)
                    print("Reach tolerance threshold. Stopping training.")
                    print("Total epoch:", epoch)
                    break

    
    def predict(self):
        self.model.eval()
        _, _,z= self.model(self.X,self.G_gene,self.row_spa,self.col_spa)
        q=get_q(self,z)
        latent_z = z.data.cpu().numpy()
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        prob=q.data.cpu().numpy()
        return y_pred,latent_z,prob