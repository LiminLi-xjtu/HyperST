

import os
import sys
import json
import random
import time

sys.path.append("../HyperST")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from config import set_arg
from utils import preprocess_data, pairwise_distance, construct_H_with_KNN_from_distance, refine_label
from ablations import Trainer, construct_gene_graph


# Parameters and random seeds
args = set_arg().parse_args([])
variants = [
    "full", "no_cross", "no_cross_mean", "gene_only", "spatial_only",
    "two_knn_graphs", "no_contrast", "no_reconstruction", "kl_only", "full_hvg",
    "simplified",
]
sample = "151509"
BASE_PATH = "../../datasets/DLPFC"
args.n_domains = 7
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = args.seed
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


# Load and preprocess data
adata = sc.read_visium(
    f"{BASE_PATH}/{sample}", count_file=f"{sample}_filtered_feature_bc_matrix.h5",
)
adata.var_names_make_unique()
adata_raw = adata.copy()
adata = preprocess_data(adata)


# Construct hypergraphs and compute PCA
if any(variant != "full_hvg" for variant in variants):
    H_gene, _ = construct_gene_graph(adata_raw.copy())
data_loc = adata.obsm["spatial"].astype(np.float32)
data_dismat = pairwise_distance(data_loc)
H_spatial = construct_H_with_KNN_from_distance(data_dismat, method="knn", k_neig=args.knn)
sc.pp.pca(adata, n_comps=args.latent_dim)

if "full_hvg" in variants:
    H_gene_hvg, _ = construct_gene_graph(adata_raw.copy(), variant="full_hvg")


# Load ground-truth labels
truth = pd.read_csv(f"{BASE_PATH}/{sample}/{sample}_truth.txt", sep="\t", header=None, index_col=0)
truth.columns = ["Ground Truth"]
adata.obs = adata.obs.join(truth)


results = []
for variant in variants:
    print(f"\n===== {variant} =====")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    adata_run = adata.copy()
    H_current = H_gene_hvg if variant == "full_hvg" else H_gene

    start = time.perf_counter()
    HyperST_model = Trainer(adata_run, H_current, H_spatial, params=args, variant=variant, init=args.init, seed=args.seed)
    HyperST_model.training_model()
    y_pred, latent_z, prob = HyperST_model.predict()
    runtime = time.perf_counter() - start
    adata_run.obs["clustering"] = pd.Categorical(y_pred)
    adata_run.obsm["HyperST"] = latent_z

    adata_run = refine_label(adata_run, key="clustering")
    valid = adata_run.obs["Ground Truth"].notna()
    labels = adata_run.obs.loc[valid, "Ground Truth"].astype(str)
    pred = adata_run.obs.loc[valid, "clustering"].astype(str)
    refined = adata_run.obs.loc[valid, "label_refined"].astype(str)
    metrics = dict(
        ari=adjusted_rand_score(labels, pred), nmi=normalized_mutual_info_score(labels, pred),
        ari_refined=adjusted_rand_score(labels, refined),
        nmi_refined=normalized_mutual_info_score(labels, refined),
    )
    print(metrics)

    output_dir = f"outputs/{variant}_{sample}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(latent_z, index=adata_run.obs_names).to_csv(f"{output_dir}/embedding.csv")
    adata_run.obs[["Ground Truth", "clustering", "label_refined"]].to_csv(f"{output_dir}/labels.csv")
    options = {k: v for k, v in vars(args).items() if k != "device"}
    options.update(data_root=BASE_PATH, cluster_seed=HyperST_model.seed, refine_neighbors=50)
    result = dict(sample=sample, variant=variant, gene_view="hvg" if variant == "full_hvg" else "svg",
                  seed=seed, param_key="tutorial", status="ok", runtime_seconds=runtime,
                  training_options=json.dumps(options, sort_keys=True), **metrics)
    pd.DataFrame([result]).to_csv(f"{output_dir}/result.csv", index=False)
    results.append(result)

    del HyperST_model, adata_run


results = pd.DataFrame(results)
results.to_csv(f"outputs/{sample}_seed{seed}_results.csv", index=False)
print(results[["variant", "ari", "ari_refined", "nmi", "nmi_refined"]])
