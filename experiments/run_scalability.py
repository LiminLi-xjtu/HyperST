#!/usr/bin/env python3

import gc
import os
import sys
from multiprocessing import get_context
from types import SimpleNamespace

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append("./HyperST")

import numpy as np
import pandas as pd
import torch
from config import set_arg
from scalability import (Trainer, ResourceMonitor, set_seed, read_dataset, preprocess,
                         gene_features, build_gene_graph, build_spatial_graph,
                         generate_G_from_H_sparse, cluster_embedding)



seeds = [2024] #, 0, 1, 100, 666
data_root = "../datasets"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = "outputs"

profiles = {
    "BRCA": dict(label="BRCA", path="BRCA", model_mode="full", spots=3798, n_domains=20,
                 init="kmeans", final_clustering="deep", preprocessing="counts",
                 spatial_mode="knn", spatial_knn=6, spatial_radius=150),
    "DLPFC": dict(label="DLPFC (151509)", path="DLPFC/151509", model_mode="full", spots=4789,
                  n_domains=7, init="mclust", final_clustering="deep", preprocessing="counts",
                  spatial_mode="knn", spatial_knn=6, spatial_radius=150),
    "MBH": dict(label="MBH", path="Slide-seqv2_Hippocampus/Hippocampus.h5ad", model_mode="simplified",
                spots=41786, n_domains=10, init="mclust", final_clustering="mclust",
                preprocessing="log_normalized", spatial_radius=40),
    "MOB": dict(label="MOB", path="Mouse_Olfactory/filtered_feature_bc_matrix.h5ad", model_mode="simplified",
                spots=19109, n_domains=7, nit="kmeans",
                final_clustering="mclust", preprocessing="counts", spatial_radius=50),
    "MSC": dict(label="MSC", path="osmFISH/osmFISH_cortex.h5ad", model_mode="full", spots=4839,
                n_domains=11, init="kmeans", final_clustering="mclust",
                preprocessing="counts", spatial_mode="knn", spatial_knn=10,
                pcs=30, hidden_dim=30, output_dim=30, n_hvg=33),
    "MVC": dict(label="MVC", path="STARmap/STARmap_20180505_BY3_1k.h5ad", model_mode="full", spots=1207,
                n_domains=7, init="mclust", final_clustering="mclust",
                preprocessing="counts", spatial_mode="knn", spatial_knn=10),
}

parser = set_arg()

parser.set_defaults(tol=-1.0, mode_knn="radius", radius=40)
base_args = parser.parse_args([])
defaults = vars(base_args).copy()
defaults.update(pcs=base_args.latent_dim, hidden_dim=base_args.hidden_dim1,
                output_dim=base_args.hidden_dim2, spatial_mode=base_args.mode_knn,
                spatial_knn=base_args.knn, spatial_radius=base_args.radius,
                n_hvg=3000, n_svg=3000, min_gene_cells=0, kmeans_n_init=20,
                gene_knn=6, elastic_gamma=1.0, elastic_beta=0.1, elastic_solver="auto"
                )

metrics = {
    "runtime_seconds": "Runtime (s)",
    "peak_gpu_allocated_mb": "Peak GPU allocated (MB)",
    "peak_gpu_reserved_mb": "Peak GPU reserved (MB)",
    "peak_cpu_memory_mb": "Peak CPU memory (MB)",
}


def summarize(results):
    rows = []
    for spec in profiles.values():
        runs = results.loc[results.dataset == spec["label"]]
        successful = runs.loc[runs.status == "ok"]
        row = {"Dataset": spec["label"], "Model variant": spec["model_mode"].title(), "Spots": spec["spots"]}
        for key, title in metrics.items():
            if runs.empty:
                row[title] = "pending"
            elif successful.empty:
                row[title] = "failed"
            elif len(successful) == 1:
                row[title] = f"{successful[key].iloc[0]:.1f} (n=1)"
            else:
                row[title] = f"{successful[key].mean():.1f} ± {successful[key].std(ddof=1):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(config, result_file):
    args = SimpleNamespace(**config)
    args.device = torch.device(args.device)
    set_seed(args.seed)
    stages = {}
    row = dict(dataset=args.label, model_variant=args.model_mode.title(), spots=args.spots,
               seed=args.seed, status="error", error="", device=str(args.device))
    overall = ResourceMonitor(args.device)
    try:
        with overall:
            source = read_dataset(args.name, args.input_path)
            adata = source.copy()
            del source
            gc.collect()
            if adata.n_obs != args.spots:
                raise ValueError(f"Expected {args.spots} spots for {args.label}, found {adata.n_obs}.")
            with ResourceMonitor(args.device, stages, "preprocess"):
                raw = adata.copy()
                adata = preprocess(adata, args)
                X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
                coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
            with ResourceMonitor(args.device, stages, "gene_graph"):
                features = gene_features(raw, args)
                H_gene = build_gene_graph(features, args)
                G_gene = generate_G_from_H_sparse(H_gene)
                A_gene = (H_gene @ H_gene.T).tocsr().astype(np.float32) if args.model_mode == "full" else None
            del raw, features
            gc.collect()
            with ResourceMonitor(args.device, stages, "spatial_graph"):
                H_spatial = build_spatial_graph(coordinates, args)
                A_spatial = (H_spatial @ H_spatial.T).tocsr().astype(np.float32) if args.model_mode == "full" else None
            with ResourceMonitor(args.device, stages, "model_setup"):
                trainer = Trainer(X, G_gene, H_spatial, A_gene, A_spatial, args)
            with ResourceMonitor(args.device, stages, "initialization"):
                trainer.initialize()
            with ResourceMonitor(args.device, stages, "training"):
                trainer.train()
            with ResourceMonitor(args.device, stages, "inference"):
                predictions, embedding = trainer.predict()
            with ResourceMonitor(args.device, stages, "clustering"):
                labels = cluster_embedding(embedding, args, args.final_clustering, predictions)
            row["status"] = "ok"
            del trainer, predictions, embedding, labels
            gc.collect()
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        print(row["error"], flush=True)
    row.update(runtime_seconds=sum(stage["seconds"] for stage in stages.values()),
               peak_gpu_allocated_mb=max((stage["peak_gpu_allocated_mb"] for stage in stages.values()), default=0),
               peak_gpu_reserved_mb=max((stage["peak_gpu_reserved_mb"] for stage in stages.values()), default=0),
               peak_cpu_memory_mb=overall.values["peak_rss_mb"])
    pd.DataFrame([row]).to_csv(result_file, index=False)


# Each repetition runs in a fresh process to isolate memory measurements.
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for name, spec in profiles.items():
        for seed in seeds:
            config = dict(defaults, **spec)
            config.update(name=name, input_path=f"{data_root}/{spec['path']}", seed=seed, device=device)
            run_dir = f"{output_dir}/{name}_seed{seed}"
            os.makedirs(run_dir, exist_ok=True)
            result_file = f"{run_dir}/result.csv"
            print(f"{spec['label']}: {spec['model_mode']}, spots={spec['spots']}, seed={seed}", flush=True)
            process = get_context("spawn").Process(target=run_dataset, args=(config, result_file))
            process.start()
            process.join()
            if process.exitcode != 0:
                failure = dict(dataset=spec["label"], model_variant=spec["model_mode"].title(),
                               spots=spec["spots"], seed=seed, status="error", device=device,
                               error=f"Process exited with code {process.exitcode}")
                failure.update({metric: np.nan for metric in metrics})
                pd.DataFrame([failure]).to_csv(result_file, index=False)
            results.append(pd.read_csv(result_file))
            frame = pd.concat(results, ignore_index=True)
            frame.to_csv(f"{output_dir}/results.csv", index=False)
            summarize(frame).to_csv(f"{output_dir}/table.csv", index=False)
    print(summarize(frame).to_string(index=False))
