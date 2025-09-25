import scanpy as sc
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from bulk_analyze import get_bulk_data, normalize_data

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='../data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad')
argparser.add_argument('--cell_type_res', type=str, default='sub_clust')
args = argparser.parse_args()

def flatten(idx_and_cell_types, bulk_data, times_sorted):
    """Given the scrna seq data and the bulk data, return the annotated flattened version"""
    print(f'Pre-flattened shape: {bulk_data.shape}')

    cell_types = [cell_type for _, cell_type in idx_and_cell_types]
    cell_type_obs = np.repeat(cell_types, len(times_sorted))

    times_obs = times_sorted * len(cell_types)

    # now let's flatten K x T x G to get (K x T) x G instead
    flattened_genes = torch.flatten(bulk_data, 0, 1)
    print(f'Post flattened shape: {flattened_genes.shape}')

    # not the best fix, but let's just replace all Nans with a 0 (should be the poor quality ones)
    flattened_genes = torch.nan_to_num(flattened_genes, nan=0.0)

    adata = sc.AnnData(flattened_genes.numpy())
    adata.obs[args.cell_type_res] = cell_type_obs
    adata.obs['numerical_age'] = times_obs
    return adata

def visualize(flattened_genes, method='umap', color='cell_type', is_bulk=False, save_to_folder='figures/'):
    # for either of them, we need to use PCA to reduce the dimensionality
    plt.figure(figsize=(20, 15))
    sc.tl.pca(flattened_genes, svd_solver='arpack')
    
    is_gaba = "gaba" in args.data_path
    folder = f'{save_to_folder}/{"gaba" if is_gaba else "all"}/{"bulk" if is_bulk else "single_cell"}'
    os.makedirs(folder, exist_ok=True)

    if method == 'umap':
        sc.pp.neighbors(flattened_genes, n_neighbors=10, n_pcs=40)
        sc.tl.umap(flattened_genes)
        sc.pl.umap(
            flattened_genes,
            title=f'U-Map of Herring Dataset ({"GABA-" if is_gaba else "All" }, {"Bulk" if is_bulk else "Single-Cell"})',
            color=color,
            legend_loc='right margin',
            legend_fontsize=6,
            legend_fontoutline=1
        )
        plt.tight_layout()        
        plt.savefig(f'{folder}/umap_{args.cell_type_res}_ann_{color}.png')

    elif method == 'tsne':
        sc.tl.tsne(flattened_genes)
        sc.pl.tsne(
            flattened_genes,
            title=f't-SNE of Herring Dataset ({"GABA-" if is_gaba else "All" }, {"Bulk" if is_bulk else "Single-Cell"})',
            color=color,
            legend_loc='right margin',
            legend_fontsize=6,
            legend_fontoutline=1
        )
        plt.tight_layout()
        plt.savefig(f'{folder}/tsne_{args.cell_type_res}_ann_{color}.png')

def visualize_cell_type_partitions(scrna_seq, idx_and_cell_types):
    print(f'There are {len(idx_and_cell_types)} types of {args.cell_type_res}.')
    is_gaba = "gaba" in args.data_path

    for _, cell_type in idx_and_cell_types:
        visualize(
            scrna_seq[scrna_seq.obs[args.cell_type_res] == cell_type],
            method='umap',
            color='numerical_age',
            is_bulk=False,
            save_to_folder=f'figures/cell_types/{"gaba" if is_gaba else "all"}/{args.cell_type_res}/{cell_type}'
        )
        print(f'Finished for cell type {cell_type}...')

scrna_seq = sc.read_h5ad(args.data_path)
bulk_data, times_sorted, idx_and_cell_types = get_bulk_data(scrna_seq, cell_type_res=args.cell_type_res)
normalized_bulk_data = normalize_data(bulk_data, idx_and_cell_types)

flattened_data = flatten(idx_and_cell_types, normalized_bulk_data, times_sorted)

visualize_cell_type_partitions(scrna_seq, idx_and_cell_types)

# visualize the single-cell data for each type by time
# visualize()

# visualizes the bulk data by cell type
# visualize(flattened_data, 'tsne', 'sub_clust', is_bulk=True)
# visualize(flattened_data, 'umap', args.cell_type_res, is_bulk=True)

# # visualizes the bulk data by time
# visualize(flattened_data, 'tsne', 'numerical_age', is_bulk=True)
# visualize(flattened_data, 'umap', 'numerical_age', is_bulk=True)

# # now, let's visualize the single cell data by cell type
# visualize(scrna_seq, 'tsne', 'sub_clust', is_bulk=False)
# visualize(scrna_seq, 'umap', args.cell_type_res, is_bulk=False)
# visualizes the single cell data by time
# visualize(scrna_seq, 'tsne', 'numerical_age', is_bulk=False)
# visualize(scrna_seq, 'umap', 'numerical_age', is_bulk=False)
