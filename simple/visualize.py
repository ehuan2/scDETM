import scanpy as sc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from bulk_analyze import get_bulk_data, normalize_data

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='../data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad')
args = argparser.parse_args()

scrna_seq = sc.read_h5ad(args.data_path)
bulk_data, times_sorted, idx_and_cell_types = get_bulk_data(scrna_seq)
normalized_bulk_data = normalize_data(bulk_data, idx_and_cell_types)

def flatten(idx_and_cell_types, bulk_data, times_sorted):
    """Given the scrna seq data and the bulk data, return the annotated flattened version"""
    print(f'Pre-flattened shape: {bulk_data.shape}')

    cell_types = [cell_type for _, cell_type in idx_and_cell_types]
    cell_type_obs = np.repeat(cell_types, len(times_sorted))

    # now let's flatten K x T x G to get (K x T) x G instead
    flattened_genes = torch.flatten(bulk_data, 0, 1)
    print(f'Post flattened shape: {flattened_genes.shape}')

    # not the best fix, but let's just replace all Nans with a 0 (should be the poor quality ones)
    flattened_genes = torch.nan_to_num(flattened_genes, nan=0.0)

    adata = sc.AnnData(flattened_genes.numpy())
    adata.obs["cell_type"] = cell_type_obs
    return adata

def visualize(flattened_genes, method='umap'):
    # for either of them, we need to use PCA to reduce the dimensionality
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    sc.tl.pca(flattened_genes, svd_solver='arpack')
    
    if method == 'umap':
        sc.pp.neighbors(flattened_genes, n_neighbors=10, n_pcs=40)
        sc.tl.umap(flattened_genes)
        sc.pl.umap(flattened_genes, title='U-Map of Herring Dataset', color='cell_type', legend_loc='upper right')
        plt.savefig(f'figures/umap_{"gaba" if "gaba" in args.data_path else "all"}.png')
    elif method == 'tsne':
        sc.tl.tsne(flattened_genes)
        sc.pl.tsne(flattened_genes, title='t-SNE of Herring Dataset', color='cell_type', legend_loc='upper right')
        plt.savefig(f'figures/tsne_{"gaba" if "gaba" in args.data_path else "all"}.png')

flattened_data = flatten(idx_and_cell_types, normalized_bulk_data, times_sorted)

visualize(flattened_data, 'tsne')
visualize(flattened_data, 'umap')
