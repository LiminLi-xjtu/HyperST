# HyperST
## Overview
HyperST: hypergraph learning with graph-guided contrastive refinement for spatial domain identification.
<img src="HyperST.jpg" width="600" />
HyperST identifies spatial domains by learning low-dimensional embeddings from two complementary hypergraphs: a spatial hypergraph constructed from spatial coordinates using kNN and an attribute hypergraph built from expression similarity over spatially variable genes. These two views are integrated through cross-fusion and an attention mechanism. In addition, HyperST can leverage standard graph structure by introducing contrastive learning between hypergraph and graph views to enhance representation quality.    Notably, for large-scale datasets, we do not recommend enabling the contrastive module due to memory constraints.

## Getting Started
### System Requirements

HyperST is implemented in **Python 3.9.19** and requires an R environment (**R 4.3.1**) for installing SPARK-related dependencies.

### 1) Create and activate a conda environment

```bash
# Create environment (Python + R)
conda create -n HyperST python=3.9.19 r-base=4.3.1

# Activate environment
conda activate HyperST

conda install r-essentials
conda install -c conda-forge pkg-config
pip install -r requirements.txt
