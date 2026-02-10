import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import random
import numba
import os
from sklearn.metrics import pairwise_distances
import cvxpy as cp
import scipy.sparse as sparse
from cvxpy.error import SolverError

### 构建邻接矩阵
@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    euclidean_dist = np.sqrt(sum)
    return euclidean_dist

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj

### HyperGraph incidence matrix (structure) 
# @numba.njit(parallel=True)
def construct_H_matrix(dis_mat, k_neig, is_probH=False, m_prob=1, method='knn', radius=None):
    """
    Construct hypergraph incidence matrix from hypergraph node distance matrix.
    :param dis_mat: Node distance matrix.
    :param k_neig: K nearest neighbors for KNN method.
    :param is_probH: Whether to construct a probabilistic Vertex-Edge matrix or binary.
    :param m_prob: Parameter for probability calculation.
    :param method: Method to construct the incidence matrix ('knn' or 'radius').
    :param radius: Radius value for radius method.
    :return: N_object X N_hyperedge incidence matrix.

    """
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        # avg_dis = np.average(dis_vec)
        
        if method == 'knn':
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx
            neighbors = nearest_idx[:k_neig]
        elif method == 'radius':
            if radius is None:
                raise ValueError("Radius value must be provided when using radius method.")
            neighbors = np.where(dis_vec <= radius)[0]
        else:
            raise ValueError("Invalid method. Choose 'knn' or 'radius'.")

        avg_dis = np.average(dis_vec) if is_probH else None

        for node_idx in neighbors:
            if is_probH:
                prob_value = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                H[node_idx, center_idx] = prob_value
                print(f"Distance: {dis_vec[node_idx]}, avg_dis: {avg_dis}, prob_value: {prob_value}")  # 调试输出
            else:
                H[node_idx, center_idx] = 1.0
    
    return H

def construct_H_with_KNN_from_distance(dis_mat, k_neig=None, is_probH=False, m_prob=1, method='knn', radius=None):
    """
    converts the result to a sparse matrix.
    """
    # Call the numba-accelerated function to get H matrix
    H = construct_H_matrix(dis_mat, k_neig, is_probH, m_prob, method,radius)
    
    # Convert to sparse matrix
    H_sparse = sp.csr_matrix(H)
    
    return H_sparse

from numpy.linalg import norm

def row_l2_normalize(X, eps=1e-12):
    """
    x / ||x||_2
    """
    X = np.asarray(X, dtype=np.float64)
    norms = norm(X, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    return X / norms

### 利用SPARK-X筛选SVGs，并构造超边
import scipy.sparse as sp
from scipy.sparse import spmatrix
def construct_H_from_attribute_matrix(adata,nSE=3000):


    adata1=adata.copy()
    # sc.pp.filter_genes(adata, max_cells=2000)

    sc.pp.filter_genes(adata,min_counts=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    import rpy2.robjects as robjects
    robjects.r.library("SPARK")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    random_seed=0
    np.random.seed(random_seed)
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rSPARKX = robjects.r['sparkx']

    # X=adata.X.toarray().T
    X = adata.X.toarray().T if isinstance(adata.X, spmatrix) else adata.X.T
    xy=adata.obsm['spatial']
    gene_names=adata.var.index.values

    sparkx_output = rSPARKX(rpy2.robjects.numpy2ri.numpy2rpy(X), rpy2.robjects.numpy2ri.numpy2rpy(xy),numCores=1) 

    adjusted_pvals = np.array(sparkx_output.rx2('res_mtest').rx2('adjustedPval'))

    top_gene_indices = np.argsort(adjusted_pvals)[:nSE]
    filtered_gene_names = gene_names[top_gene_indices]

    # X1=adata.X.toarray()
    X1 = adata.X.toarray() if isinstance(adata.X, spmatrix) else adata.X
    X_preprocess = X1[:, top_gene_indices]

    filtered_adata = adata1[:, filtered_gene_names].copy() 
    sc.pp.normalize_total(filtered_adata, target_sum=1e4)
    sc.pp.scale(filtered_adata)

    X_111=filtered_adata.X

    # X_222=row_l2_normalize(X_111)
    H=gen_l1_l2_hg(X_111,gamma=1.0,beta=0.1,n_neighbors=6)

    rpy2.robjects.numpy2ri.deactivate()
    return H,X_111



def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='embed', random_seed=2020):

    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def mclust(data, num_cluster, modelNames = 'EEE', random_seed = 2020):
    """
    Mclust algorithm from R, similar to https://mclust-org.github.io/mclust/
    """ 
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()  
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(data, num_cluster, modelNames)
    return np.array(res[-2])


import ot

def refine_label(adata, radius=50, key='clustering'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    # adata.obs['label_refined'] = np.array(new_type)

    unique_old_types = set(old_type)  
    unique_new_types = set(new_type)  
    
    if len(unique_old_types) != len(unique_new_types):
        print("Warning: Refine process changed the number of unique types. Returning original labels.")
        adata.obs['label_refined'] = old_type  
    else:
        adata.obs['label_refined'] = np.array(new_type)  
 
    return adata



def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC


def preprocess_data(adata):
    # sc.pp.filter_genes(adata,min_counts=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var['highly_variable']].copy()

    return adata

import numpy as np
from sklearn.metrics import pairwise_distances
import cvxpy as cp
import scipy.sparse as sparse
from cvxpy.error import SolverError
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

def pairwise_geographic_similarity(X):
    # 计算成对的欧式距离矩阵
    euclidean_dist_matrix = cdist(X, X, metric='euclidean')
    # 将欧式距离矩阵转换为地理相似性矩阵
    similarity_matrix = 1.0 / (1.0 + euclidean_dist_matrix)
    return similarity_matrix

# Function with L1 and L2 regularization
def gen_l1_l2_hg(X,gamma, beta,n_neighbors):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param gamma: float, the tradeoff parameter of the l1 norm on representation coefficients
    :param beta: float, the tradeoff parameter of the l2 norm (regularization)
    :param n_neighbors: int,
    :param log: bool
    :param with_feature: bool, optional(default=False)
    :return: instance of HyperG
    """

    assert n_neighbors >= 1.
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X)
    m_neighbors = np.argsort(m_dist)[:, 0:n_neighbors+1]

    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)
    node_idx = []
    values = []

    for i_edge in range(n_edges):

        neighbors = m_neighbors[i_edge].tolist()
        if i_edge in neighbors:
            neighbors.remove(i_edge)
        else:
            neighbors = neighbors[:-1]

        P = X[neighbors, :]
        v = X[i_edge, :]

        # cvxpy
        x = cp.Variable(P.shape[0], nonneg=True)
        # constraints1 = [cp.sum(x) == 1]

        # Define the objective with L1 and L2 regularization
        objective = cp.Minimize(
            cp.norm((P.T @ x).T - v, 2) + 
            gamma * cp.norm(x, 1)+   # L1 regularization for sparsity
            beta * cp.norm(x, 2)     # L2 regularization for smoothness
        )

        prob = cp.Problem(objective)
        try:
            prob.solve()
        except SolverError:
            prob.solve(solver='SCS', verbose=False)

        node_idx.extend([i_edge] + neighbors)
        values.extend([1.] + x.value.tolist())

    node_idx = np.array(node_idx)
    values = np.array(values)

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    return H