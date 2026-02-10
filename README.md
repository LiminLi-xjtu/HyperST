# HyperST
## Overview
HyperST: hypergraph learning with graph-guided contrastive refinement for spatial domain identification.
<img src="HyperST.jpg" width="600" />
HyperST identifies spatial domains by learning low-dimensional embeddings from two complementary hypergraphs: a spatial hypergraph constructed from spatial coordinates using kNN and an attribute hypergraph built from expression similarity over spatially variable genes. These two views are integrated through cross-fusion and an attention mechanism. In addition, HyperST can leverage standard graph structure by introducing contrastive learning between hypergraph and graph views to enhance representation quality.    Notably, for large-scale datasets, we do not recommend enabling the contrastive module due to memory constraints.

## Getting Started
### System Requirements

HyperST is implemented in **Python 3.9.19** and requires an R environment (**R 4.3.1**) for installing SPARK-X related dependencies.

### 1) Create and activate a conda environment

```bash
# Create environment (Python + R)
conda create -n HyperST python=3.9.19 r-base=4.3.1

# Activate environment
conda activate HyperST
```

### 2) Install dependencies

```bash
conda install r-essentials
conda install -c conda-forge pkg-config
pip install -r requirements.txt
```

### 3) (Optional) Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name HyperST
```

### 4) Install required R packages

> **Note:** Run the following commands in the R console.

```r
install.packages("devtools")
install.packages("remotes", repos = "[https://cloud.r-project.org](https://cloud.r-project.org)")

# Install mclust
remotes::install_version("mclust", version = "6.1.1", repos = "[https://cloud.r-project.org](https://cloud.r-project.org)")

# Install SPARK from GitHub
devtools::install_github("xzhoulab/SPARK")
```

## Tutorial

For a detailed demonstration of **HyperST**, please refer to the tutorial notebook:

* [**DLPFC_Tutorial.ipynb**](./DLPFC_Tutorial.ipynb)

This notebook provides a step-by-step guide on how to run the model.
