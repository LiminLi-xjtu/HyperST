import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from layers import HGNN_conv, UniGATConv, GraphConvolution
from model import HyperST, HyperST_sparse
from model_train import (
    HyperSTModel_Train, get_q, target_distribution, iterative_loss,
    info_nce_loss, reconstruction_loss,
)
from utils import construct_H_from_attribute_matrix, gen_l1_l2_hg, mclust


class NoCrossModel(HyperST):
    def fuse(self, out1, out2):
        embed, _ = self.att_1(torch.stack([out2, out1], dim=1))
        return embed

    def forward(self, X_gene, G_gene, vertex_spa, edges_spa, A_gene, A_spatial):
        H1 = self.conv1_1(X_gene, G_gene)
        H2 = self.conv1_2(X_gene, vertex_spa, edges_spa)
        out1 = self.conv2_1(H1, G_gene)
        out2 = self.conv2_2(H2, vertex_spa, edges_spa)
        embed = self.fuse(out1, out2)
        h1 = F.relu(self.conv1_3(X_gene, A_gene))
        s1 = F.relu(self.conv1_4(X_gene, A_spatial))
        h2 = self.conv2_3(h1, A_gene)
        s2 = self.conv2_4(s1, A_spatial)
        X_dec = self.decoder1(h2, A_gene)
        return h2, s2, embed, X_dec, out1, out2


class NoCrossMeanModel(NoCrossModel):
    def fuse(self, out1, out2):
        return (out1 + out2) / 2


class GeneOnlyModel(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = HGNN_conv(in_channels, hidden_channels)
        self.conv2 = HGNN_conv(hidden_channels, out_channels)
        self.graph1 = GraphConvolution(in_channels, hidden_channels)
        self.graph2 = GraphConvolution(hidden_channels, out_channels)
        self.decoder = GraphConvolution(out_channels, in_channels)

    def forward(self, X_gene, G_gene, vertex_spa, edges_spa, A_gene, A_spatial):
        embed = self.conv2(self.conv1(X_gene, G_gene), G_gene)
        h2 = self.graph2(F.relu(self.graph1(X_gene, A_gene)), A_gene)
        X_dec = self.decoder(h2, A_gene)
        zeros = torch.zeros_like(h2)
        return h2, zeros, embed, X_dec, embed, zeros


class SpatialOnlyModel(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = UniGATConv(args, in_channels, hidden_channels)
        self.conv2 = UniGATConv(args, hidden_channels, out_channels)
        self.graph1 = GraphConvolution(in_channels, hidden_channels)
        self.graph2 = GraphConvolution(hidden_channels, out_channels)

    def forward(self, X_gene, G_gene, vertex_spa, edges_spa, A_gene, A_spatial):
        H1 = self.conv1(X_gene, vertex_spa, edges_spa)
        embed = self.conv2(H1, vertex_spa, edges_spa)
        s2 = self.graph2(F.relu(self.graph1(X_gene, A_spatial)), A_spatial)
        zeros = torch.zeros_like(s2)
        return zeros, s2, embed, torch.zeros_like(X_gene), zeros, embed


class TwoKNNGraphsModel(HyperST):
    def __init__(self, args, in_channels, out_channels, hidden_channels):
        super().__init__(args, in_channels, out_channels, hidden_channels)
        self.conv1_1 = GraphConvolution(in_channels, hidden_channels)
        self.conv1_2 = GraphConvolution(in_channels, hidden_channels)
        self.conv2_1 = GraphConvolution(hidden_channels, out_channels)
        self.conv2_2 = GraphConvolution(hidden_channels, out_channels)

    def forward(self, X_gene, G_gene, vertex_spa, edges_spa, A_gene, A_spatial):
        H1 = self.conv1_1(X_gene, A_gene)
        H2 = self.conv1_2(X_gene, A_spatial)
        H3, _ = self.att(torch.stack([H1, H2], dim=1))
        out1 = self.conv2_1(H3, A_gene)
        out2 = self.conv2_2(H1, A_spatial)
        embed, _ = self.att_1(torch.stack([out2, out1], dim=1))
        h1 = F.relu(self.conv1_3(X_gene, A_gene))
        s1 = F.relu(self.conv1_4(X_gene, A_spatial))
        h2 = self.conv2_3(h1, A_gene)
        s2 = self.conv2_4(s1, A_spatial)
        X_dec = self.decoder1(h2, A_gene)
        return h2, s2, embed, X_dec, out1, out2


class SimplifiedModel(HyperST_sparse):
    def forward(self, X_gene, G_gene, vertex_spa, edges_spa, A_gene, A_spatial):
        out1, out2, embed = super().forward(X_gene, G_gene, vertex_spa, edges_spa)
        return out1, out2, embed, X_gene, out1, out2


def pairwise_graph(H):
    A = H.tocsr().copy()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    A = A.maximum(A.T) + sparse.eye(A.shape[0])
    D = sparse.diags(np.asarray(A.sum(axis=1)).ravel() ** -0.5)
    return torch.tensor((D @ A @ D).toarray(), dtype=torch.float32)


def construct_gene_graph(adata, variant="full", nSE=3000):
    if variant != "full_hvg":
        return construct_H_from_attribute_matrix(adata, nSE=nSE)
    adata = adata.copy()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=nSE)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.scale(adata)
    X_hvg = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    H_gene = gen_l1_l2_hg(X_hvg, gamma=1.0, beta=0.1, n_neighbors=6)
    return H_gene, X_hvg


class Trainer(HyperSTModel_Train):
    def __init__(self, adata, H_gene, H_spatial, params, variant="full", init="mclust", seed=818):
        super().__init__(adata, H_gene, H_spatial, params, init=init, seed=seed)
        self.variant = variant
        models = {
            "no_cross": NoCrossModel,
            "no_cross_mean": NoCrossMeanModel,
            "gene_only": GeneOnlyModel,
            "spatial_only": SpatialOnlyModel,
            "two_knn_graphs": TwoKNNGraphsModel,
            "simplified": SimplifiedModel,
        }
        if variant in models:
            self.model = models[variant](
                params, params.latent_dim, params.hidden_dim2, params.hidden_dim1,
            ).to(self.device)
        if variant == "two_knn_graphs":
            self.A_gene = pairwise_graph(H_gene).to(self.device)
            self.A_spatial = pairwise_graph(H_spatial).to(self.device)

    def loss(self, z_gene, z_spa, embed, X_dec, p, q):
        loss = iterative_loss(p, q)
        if self.variant not in ("no_contrast", "kl_only", "simplified", "gene_only"):
            loss = loss + self.params.lambda_3 * info_nce_loss(z_spa, embed)
        if self.variant not in ("no_contrast", "kl_only", "simplified", "spatial_only"):
            loss = loss + self.params.lambda_2 * info_nce_loss(z_gene, embed)
        if self.variant not in ("no_reconstruction", "kl_only", "simplified", "spatial_only"):
            loss = loss + self.params.lambda_1 * reconstruction_loss(X_dec, self.X)
        return loss

    def training_model(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay,
        )
        with torch.no_grad():
            _, _, features, _, _, _ = self.model(
                self.X, self.G_gene, self.row_spa, self.col_spa, self.A_gene, self.A_spatial,
            )
        features = features.cpu().numpy()
        if self.init == "kmeans":
            y_pred = KMeans(self.n_clusters, n_init=20).fit_predict(features)
        elif self.init == "mclust":
            y_pred = mclust(features, num_cluster=self.n_clusters, random_seed=self.seed).astype(int)
        centers = pd.DataFrame(features).groupby(y_pred).mean().to_numpy()
        self.mu = torch.nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=self.device))
        y_pred_last = y_pred

        progress = tqdm(range(self.max_epochs), desc="Epoch")
        for epoch in progress:
            self.model.train()
            if epoch % self.params.update_interval == 0:
                with torch.no_grad():
                    _, _, embed, _, _, _ = self.model(
                        self.X, self.G_gene, self.row_spa, self.col_spa, self.A_gene, self.A_spatial,
                    )
                    p = target_distribution(get_q(self, embed).detach())

            z_gene, z_spa, embed, X_dec, _, _ = self.model(
                self.X, self.G_gene, self.row_spa, self.col_spa, self.A_gene, self.A_spatial,
            )
            q = get_q(self, embed)
            loss = self.loss(z_gene, z_spa, embed, X_dec, p, q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress.set_postfix(loss=loss.item())

            y_pred = q.detach().cpu().numpy().argmax(1)
            delta_label = np.mean(y_pred != y_pred_last)
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % self.params.update_interval == 0 and delta_label < self.params.tol:
                break
