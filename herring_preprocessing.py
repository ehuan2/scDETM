import scanpy as sc
import anndata
import torch
import logging
from tqdm import tqdm

# logging setup
logging.basicConfig(level=logging.DEBUG)

ann_data = sc.read_h5ad(
    f'./data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad'
)

def get_query_copy(query):
    return ann_data[ann_data.obs.query(query).index].copy()

# Example on getting male only data:
# ann_data_male = get_query_copy("Sex == 'M'")

train_times = ann_data.obs['numerical_age'].unique()

device = torch.device('cpu')

def get_rnn_input(data, batch_size=1000):
    """
    Given some annotated data, return the count of each gene per time
    in a times x genes PyTorch tensor.
    """
    num_docs = ann_data.X.shape[0] # number of cells
    num_genes = ann_data.X.shape[1] # number of genes
    # first off, split the data off into random batches
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, batch_size)

    # then, we create a times by genes data count based on sorted order
    times_sorted = sorted(ann_data.obs['numerical_age'].unique().tolist())
    rnn_input = torch.zeros(len(times_sorted), num_genes).to(device)

    for ind in tqdm(indices):
        # then we grab the gene and times data corresponding to the indices
        ind = ind.tolist()
        data_batch = torch.tensor(data.X[ind, :].toarray())
        times_batch = torch.tensor(data.obs['numerical_age'].iloc[ind])

        # then for each time step we add up the gene counts
        for t in range(len(times_sorted)):
            # get the batch indices which correspond to this time step
            batch_idxs = (times_batch == times_sorted[t]).nonzero()
            if batch_idxs.shape[0] == 0:
                continue

            # then we add up the gene counts for this time step
            gene_counts = None
            if batch_idxs.shape[0] == 1: # if it's 2D and only one of them
                gene_counts = data_batch[batch_idxs].squeeze()
            else:
                gene_counts = data_batch[batch_idxs].squeeze().sum(0)

            assert gene_counts.shape == rnn_input[t].shape
            rnn_input[t] += gene_counts
        
    # normalize the data for RNN
    rnn_input = rnn_input / rnn_input.sum(1, keepdim=True)
    return rnn_input, times_sorted

rnn_input, times_sorted = get_rnn_input(ann_data)
logging.debug(rnn_input)
