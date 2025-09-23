import scanpy as sc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

scrna_seq = sc.read_h5ad(
    f'../data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad'
)

def get_query_copy(query):
    return scrna_seq[scrna_seq.obs.query(query).index].copy()

# Example on getting male only data:
# scrna_seq_male = get_query_copy("Sex == 'M'")

train_times = scrna_seq.obs['numerical_age'].unique()

# make a pseudo-bulk data for this
# to do this, we need to first split into the different cell types
# for each different time point
# then for each cell type at a certain time point, we sum up the counts
def get_bulk_data(data):
    """
    Given some annotated data, return the bulk data for each cell type
    at each time point.
    """
    idx_and_cell_types = list(enumerate(data.obs['cell_type'].unique().tolist()))
    times_sorted = sorted(data.obs['numerical_age'].unique().tolist())
    num_genes = data.X.shape[1] # X is cells by genes
    
    bulk_data = torch.zeros(len(idx_and_cell_types), len(times_sorted), num_genes)
    
    # now we go through all the times and cell types to add up the counts
    for t_idx, t in enumerate(times_sorted):
        data_at_t = data[data.obs['numerical_age'] == t]
        # for this specific time point, we iterate over the cell types
        # and add up the counts
        for idx, cell_type in idx_and_cell_types:
            bulk_data[idx][t_idx] = torch.tensor(
                data_at_t[data_at_t.obs['cell_type'] == cell_type].X.toarray()
            ).sum(0)
    
    return bulk_data, times_sorted, idx_and_cell_types

bulk_data, times_sorted, idx_and_cell_types = get_bulk_data(scrna_seq)
normalized_bulk_data = torch.zeros_like(bulk_data)
for idx, _ in idx_and_cell_types:
    normalized_bulk_data[idx] = bulk_data[idx] / bulk_data[idx].sum(1, keepdim=True)
print(normalized_bulk_data.shape)

def heatmap(data, cell_idx, celltype):
    plt.figure(figsize=(10, 8))
    non_zero = data[cell_idx][:, data[cell_idx].sum(0) > 0.02]
    sns.heatmap(non_zero.numpy(), cmap='viridis')
    plt.title(f'Gene expression proportion for cell type over time: {celltype}')
    plt.xlabel('Genes')
    plt.ylabel('Time Points')

    x_labels = np.arange(data[cell_idx].shape[1])[np.where(data[cell_idx].sum(0) > 0.02)[0]]

    # add time labels
    plt.xticks(
        ticks=np.arange(len(x_labels))+0.5,
        labels=x_labels,
        rotation=90,
    )
    plt.yticks(
        ticks=np.arange(len(times_sorted))+0.5,
        labels=[round(t, 3) for t in times_sorted],
        rotation=0
    )
    plt.savefig(f'./figures/heatmap_{celltype}.png')
    print(f'Saved figure for heatmap_{celltype}.png')

for idx, cell_type in idx_and_cell_types:
    heatmap(normalized_bulk_data, idx, cell_type)
