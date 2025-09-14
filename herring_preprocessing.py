import scanpy as sc
import anndata
import torch
import logging

# logging setup
logging.basicConfig(level=logging.DEBUG)

ann_data = sc.read_h5ad(
    f'./data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad'
)

print(ann_data)
print(f'Shape of cells x genes sample is: {ann_data.X.shape}')
# ** can verify with the data below for if the datasets match their annotations**
# print(ann_data.obs['batch'][0])
# print(ann_data.var['gene_ids'][0])

def get_query_copy(query):
    return ann_data[ann_data.obs.query(query).index].copy()

ann_data_male = get_query_copy("Sex == 'M'")
ann_data_female = get_query_copy("Sex == 'F'")

# print(f'Male counts: {ann_data_male.shape}, Female counts: {ann_data_female}')

# print(ann_data_male.obs['numerical_age'].unique())
# print(ann_data_female.obs['numerical_age'].unique())

train_times = ann_data.obs['numerical_age'].unique()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

def get_rnn_input(data, batch_size=100):
    """
    Given some annotated data, return the count of each gene per time
    in a times x genes PyTorch tensor.
    """
    # first off, split the data off into random batches
    num_docs = ann_data.X.shape[0] # number of cells
    num_genes = ann_data.X.shape[1] # number of genes
    # first, get a random permutation over all the cells
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, batch_size)
    indices = indices[0:2]

    times_sorted = sorted(ann_data.obs['numerical_age'].unique().tolist())

    # then, we create a times by genes data count
    rnn_input = torch.zeros(len(times_sorted), num_genes).to(device)

    for idx, ind in enumerate(indices):
        # we grab the batch:
        # data_batch represents the cells x genes for this batch
        # times_batch represents the timestamp per batch index
        ind = ind.tolist()
        data_batch = torch.tensor(data.X[ind, :].toarray())
        times_batch = torch.tensor(data.obs['numerical_age'].iloc[ind])
        logging.debug(f'data batch: {data_batch}, times batch: {times_batch}')

        # then for each time step
        for t in range(len(times_sorted)):
            # tmp represents the list of indices of data_batch which match the specific time
            batch_idxs = (times_batch == times_sorted[t]).nonzero()
            if batch_idxs.shape[0] == 0:
                continue
                
            logging.debug(f'Indices of data_batch: {batch_idxs}, the specific time: {times_sorted[t]}')
            logging.debug(f'The indices of X: {[ind[batch_idx[0]] for batch_idx in batch_idxs.tolist()]}')

            to_add = None
            if batch_idxs.shape[0] == 1: # if it's 2D and only one of them
                to_add = data_batch[batch_idxs].squeeze()
            else:
                to_add = data_batch[batch_idxs].squeeze().sum(0)

            assert to_add.shape == rnn_input[t].shape
            rnn_input[t] += to_add
            logging.debug(f'Gene data of the batch: {data_batch[batch_idxs]}')

            first_non_zero_idx = rnn_input[t].nonzero()[0]
            logging.debug(f'Non-zero index of genes: {first_non_zero_idx}')
            logging.debug(f'Gene count for gene {first_non_zero_idx.item()}: {rnn_input[t, first_non_zero_idx.item()]}')
            for batch_idx in batch_idxs.tolist():
                logging.debug(f'Gene count for cell {ind[batch_idx[0]]}, gene {first_non_zero_idx.item()}: {data.X[ind[batch_idx[0]], first_non_zero_idx.item()]} or {data_batch[batch_idx[0], first_non_zero_idx.item()]}')

            over_ten_idx = (rnn_input[t] > 10).nonzero()[0]
            logging.debug(f'Expression over 10 index of genes: {over_ten_idx}')
            logging.debug(f'Gene count for gene {over_ten_idx.item()}: {rnn_input[t, over_ten_idx.item()]}')
            for batch_idx in batch_idxs.tolist():
                logging.debug(f'Gene count for cell {ind[batch_idx[0]]}, gene {over_ten_idx.item()}: {data.X[ind[batch_idx[0]], over_ten_idx.item()]} or {data_batch[batch_idx[0], over_ten_idx.item()]}')

    rnn_input = rnn_input / rnn_input.sum(1, keepdim=True)
    return rnn_input, times_sorted

logging.debug('Calling get_rnn_input function...')
rnn_input, times_sorted = get_rnn_input(ann_data)

logging.debug(f'RNN verify: {rnn_input.sum(1)}')
