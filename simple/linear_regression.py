import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from bulk_analyze import get_bulk_data, normalize_data
import networkx as nx
import fastcluster

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='../data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad')
args = argparser.parse_args()

scrna_seq = sc.read_h5ad(args.data_path)
bulk_data, times_sorted, idx_and_cell_types = get_bulk_data(scrna_seq)
normalized_bulk_data = normalize_data(bulk_data, idx_and_cell_types)

print(f'Starting linear regression')

# X is the matrix s.t. t1 ~ t0 * X
for i in range(len(times_sorted) - 1):
    # TODO: filter by expression s.t. we only look at the columns and rows
    # which are higher than some threshold in the normalized bulk data    

    t_i = normalized_bulk_data[:, i, :]
    t_i_plus_1 = normalized_bulk_data[:, i + 1, :]
    X_lstsq, residuals, rank, s = np.linalg.lstsq(t_i.numpy(), t_i_plus_1.numpy())

    t_i[:, (t_i > 0.2)]

    row_linkage = fastcluster.linkage_vector(X_lstsq, method='ward')
    print(f'Finish row linkage')
    sns.clustermap(
        X_lstsq,
        row_linkage=row_linkage,
        col_cluster=False,
        cmap='viridis',
        figsize=(8, 8)
    )

    plt.savefig(
        f'figures/X_lstsq_age_{round(times_sorted[i], 3)}_to_age_{round(times_sorted[i + 1], 3)}.png'
    )

    print(f'Saved for time step {i} to {i + 1}')
