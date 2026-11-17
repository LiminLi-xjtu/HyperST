"""Full/Simplified training and resource measurement for the six-dataset table."""

import gc
import os
import random
import threading
import time

import numpy as np
import pandas as pd
import psutil
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm

import sys
sys.path.append("./HyperST")
from model import HyperST, HyperST_sparse
from model_train import get_q, target_distribution, iterative_loss, info_nce_loss, reconstruction_loss
from utils import mclust


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_dataset(name, path):
    if name in ["DLPFC", "BRCA"]:
        sample = "151509" if name == "DLPFC" else "BRCA"
        adata = sc.read_visium(path, count_file=f"{sample}_filtered_feature_bc_matrix.h5")
    else:
        adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    if name == "MSC":
        adata = adata[adata.obs["Region"].astype(str) != "Excluded"].copy()
    return adata


def preprocess(adata, args):
    if args.min_gene_cells:
        sc.pp.filter_genes(adata, min_cells=args.min_gene_cells)
    flavor = "seurat_v3" if args.preprocessing == "counts" else "seurat"
    sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=min(args.n_hvg, adata.n_vars))
    if args.preprocessing == "counts":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.pca(adata, n_comps=args.pcs, random_state=0)
    return adata


def gene_features(raw, args):
    # Select SVGs with SPARK-X, retaining the historical preprocessing order.
    import rpy2.robjects as ro
    from rpy2.robjects.numpy2ri import numpy2rpy
    selected = raw.copy()
    sc.pp.filter_genes(selected, min_counts=5)
    sc.pp.normalize_total(selected, target_sum=1e4)
    sc.pp.log1p(selected)
    ro.r.library("SPARK")
    np.random.seed(0)
    random.seed(0)
    ro.r["set.seed"](0)
    X = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    result = ro.r["sparkx"](numpy2rpy(X.T), numpy2rpy(np.asarray(selected.obsm["spatial"])), numCores=1)
    pvalues = np.asarray(result.rx2("res_mtest").rx2("adjustedPval"))
    genes = selected.var_names[np.argsort(pvalues)[:args.n_svg]]
    selected = raw[:, genes].copy()
    sc.pp.normalize_total(selected, target_sum=1e4)
    sc.pp.scale(selected)
    X = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    return X.astype(np.float32)


def build_gene_graph(features, args):
    # NearestNeighbors avoids allocating a dense N-by-N distance matrix.
    X = np.asarray(features, dtype=np.float64)
    n, k = X.shape[0], args.gene_knn
    candidates = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    neighbors = np.vstack([row[row != i][:k] for i, row in enumerate(candidates)])
    coefficients = np.empty((n, k), dtype=np.float32)
    import cvxpy as cp
    for i in range(n):
        weights = cp.Variable(k, nonneg=True)
        objective = cp.norm(X[neighbors[i]].T @ weights - X[i], 2)
        objective += args.elastic_gamma * cp.norm(weights, 1) + args.elastic_beta * cp.norm(weights, 2)
        problem = cp.Problem(cp.Minimize(objective))
        options = {} if args.elastic_solver == "auto" else {"solver": args.elastic_solver}
        try:
            problem.solve(warm_start=True, verbose=False, **options)
        except cp.error.SolverError:
            problem.solve(solver="SCS", warm_start=True, verbose=False)
        if weights.value is None:
            raise RuntimeError(f"Elastic Net solver failed: {problem.status}")
        coefficients[i] = weights.value
    rows = np.concatenate([np.arange(n), neighbors.ravel()])
    cols = np.concatenate([np.arange(n), np.repeat(np.arange(n), k)])
    values = np.concatenate([np.ones(n, dtype=np.float32), coefficients.ravel()])
    H = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    H.eliminate_zeros()
    return H


def build_spatial_graph(coordinates, args):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    n = len(coordinates)
    if args.spatial_mode == "knn":
        model = NearestNeighbors(n_neighbors=args.spatial_knn).fit(coordinates)
        neighbors = model.kneighbors(coordinates, return_distance=False)
    else:
        model = NearestNeighbors(radius=args.spatial_radius).fit(coordinates)
        neighbors = model.radius_neighbors(coordinates, return_distance=False)
    rows = np.concatenate(neighbors)
    cols = np.repeat(np.arange(n), [len(row) for row in neighbors])
    H = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    return H.maximum(sparse.eye(n, dtype=np.float32, format="csr"))


def generate_G_from_H_sparse(H):
    H = H.tocsr().astype(np.float32)
    dv = np.asarray(H.sum(axis=1)).ravel()
    de = np.asarray(H.sum(axis=0)).ravel()
    inv_dv = np.divide(1, np.sqrt(dv), out=np.zeros_like(dv), where=dv > 0)
    inv_de = np.divide(1, de, out=np.zeros_like(de), where=de > 0)
    Dv, De = sparse.diags(inv_dv), sparse.diags(inv_de)
    return (Dv @ H @ De @ H.T @ Dv).tocsr().astype(np.float32)


def to_tensor(matrix, device):
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(coo.data), coo.shape,
                                   check_invariants=False).coalesce().to(device)


def cluster_embedding(embedding, args, method, predictions=None):
    if method == "deep":
        return predictions
    if method == "mclust":
        return mclust(embedding, args.n_domains, random_seed=args.seed).astype(int)
    return KMeans(args.n_domains, n_init=args.kmeans_n_init, random_state=args.seed).fit_predict(embedding)


class Trainer:
    def __init__(self, X, G_gene, H_spatial, A_gene, A_spatial, args):
        self.args, self.variant = args, args.model_mode
        self.beta = 0.5
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=args.device)
        self.G_gene = to_tensor(G_gene, args.device)
        coo = H_spatial.tocoo()
        self.row = torch.tensor(coo.row, dtype=torch.long, device=args.device)
        self.col = torch.tensor(coo.col, dtype=torch.long, device=args.device)
        if self.variant == "simplified":
            self.model = HyperST_sparse(args, args.pcs, args.output_dim, args.hidden_dim).to(args.device)
        else:
            self.A_gene = to_tensor(A_gene, args.device)
            self.A_spatial = to_tensor(A_spatial, args.device)
            self.model = HyperST(args, args.pcs, args.output_dim, args.hidden_dim).to(args.device)

    def forward(self):
        if self.variant == "simplified":
            return self.model(self.X, self.G_gene, self.row, self.col)
        return self.model(self.X, self.G_gene, self.row, self.col, self.A_gene, self.A_spatial)

    def initialize(self):
        self.model.eval()
        with torch.no_grad():
            features = self.forward()[2].cpu().numpy()
        labels = cluster_embedding(features, self.args, self.args.init)
        labels = pd.factorize(labels, sort=True)[0]
        centers = np.vstack([features[labels == i].mean(axis=0) for i in range(self.args.n_domains)])
        self.mu = torch.nn.Parameter(torch.tensor(centers, device=self.args.device), requires_grad=False)
        parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def train(self):
        previous = None
        for epoch in tqdm(range(self.args.epochs), desc=self.variant):
            self.model.train()
            if epoch % self.args.update_interval == 0:
                with torch.no_grad():
                    p = target_distribution(get_q(self, self.forward()[2]).detach())
            output = self.forward()
            embedding = output[2]
            q = get_q(self, embedding)
            kl = iterative_loss(p, q)
            spatial = gene = reconstruction = kl.new_tensor(float("nan"))
            loss = kl
            if self.variant == "full":
                spatial = info_nce_loss(output[1], embedding)
                gene = info_nce_loss(output[0], embedding)
                reconstruction = reconstruction_loss(output[3], self.X)
                # Preserve historical meanings: lambda_1=spatial, lambda_2=gene, lambda_3=reconstruction.
                loss = loss + self.args.lambda_1 * spatial + self.args.lambda_2 * gene + self.args.lambda_3 * reconstruction
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            current = q.detach().cpu().numpy().argmax(1)
            if previous is not None and epoch > 0 and (epoch - 1) % self.args.update_interval == 0:
                if np.mean(current != previous) < self.args.tol:
                    break
            previous = current

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            embedding = self.forward()[2]
            q = get_q(self, embedding)
        return q.argmax(1).cpu().numpy(), embedding.cpu().numpy()


class ResourceMonitor:
    def __init__(self, device, records=None, name="run"):
        self.device, self.records, self.name = device, records, name
        self.process = psutil.Process()
        self.stop = threading.Event()

    def sample(self):
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_rss = self.peak = self.process.memory_info().rss
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        self.values = dict(seconds=elapsed, peak_rss_mb=self.peak / 1024**2,
                           rss_delta_mb=max(0, self.peak - self.start_rss) / 1024**2,
                           peak_gpu_allocated_mb=0.0, peak_gpu_reserved_mb=0.0)
        if self.device.type == "cuda":
            self.values.update(peak_gpu_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
                               peak_gpu_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2)
        if self.records is not None:
            self.records[self.name] = self.values
