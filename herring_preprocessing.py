import scanpy as sc
import torch
import logging
from tqdm import tqdm
from main import get_args, get_optimizer
from detm import DETM, device
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# logging setup, to the current timestamp
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(
    filename=f'./logs/debug/{current_time}.log',
    level=logging.DEBUG
)

ann_data = sc.read_h5ad(
    f'./data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad'
)

def get_query_copy(query):
    return ann_data[ann_data.obs.query(query).index].copy()

# Example on getting male only data:
# ann_data_male = get_query_copy("Sex == 'M'")

train_times = ann_data.obs['numerical_age'].unique()

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
    rnn_input = torch.zeros(len(times_sorted), num_genes)

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

def get_cell_types(data):
    """
    Given some annotated data, return the unique cell types.
    """
    return data.obs['cell_type'].unique().tolist()

# * TODO: get the model working with this data
# [x] Get RNN input
# [ ] Recreate main.py's train() function

# first grab the args from the command line
args = get_args()
rnn_input, times_sorted = get_rnn_input(ann_data)

# then set the args appropriately
args.vocab_size = rnn_input.shape[1]
args.num_times = len(times_sorted)
args.train_embeddings = True # we will probably want to train the embeddings instead
embeddings = None

# set the number of topics to be the number of cell types
cell_types = get_cell_types(ann_data)
logging.debug(f'Cell types: {cell_types}')
args.num_topics = 10
logging.debug(f'Number of topics: {args.num_topics}')

logging.debug(f'Number of cells: {ann_data.X.shape[0]}, batch size: {args.batch_size}')
args.batch_size = 32

model = DETM(args, embeddings).to(device)
logging.debug(model)

# then, let's get the optimizer
optimizer = get_optimizer(model, args)

writer = SummaryWriter(f'./logs/scDETM_runs')

def train(model, data, optimizer, rnn_input, times_sorted, epoch):
    """Train DETM on data for one epoch."""
    model.train()
    rnn_input = rnn_input.to(device)

    # set the total loss, cross-entropy, and KL losses
    total_loss = 0
    total_nll = 0
    total_kl_theta_loss = 0
    total_kl_eta_loss = 0
    total_kl_alpha_loss = 0

    # batchify the data
    total_cells = ann_data.X.shape[0]
    indices = torch.randperm(total_cells)
    indices = torch.split(indices, args.batch_size) 

    for step, ind in tqdm(enumerate(indices)):
        optimizer.zero_grad()
        model.zero_grad()

        ind = ind.tolist()
        data_batch = torch.tensor(data.X[ind, :].toarray()).to(device)

        # create a times batch which represents the time point index
        # in the sorted list of time points for each cell in the batch
        times_batch = torch.zeros((len(ind), )).to(device)
        for i, idx in enumerate(ind):
            time_point = (
                torch.tensor(times_sorted) == data.obs["numerical_age"].iloc[idx]
            ).nonzero().squeeze()
            times_batch[i] = time_point
        
        sums = data_batch.sum(1, keepdim=True)

        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        loss, nll, kl_alpha, kl_eta, kl_theta = model(
            data_batch,
            normalized_data_batch,
            times_batch,
            rnn_input,
            total_cells
        )

        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += torch.sum(loss).item()
        total_nll += torch.sum(nll).item()
        total_kl_theta_loss += torch.sum(kl_theta).item()
        total_kl_eta_loss += torch.sum(kl_eta).item()
        total_kl_alpha_loss += torch.sum(kl_alpha).item()

        if step % args.log_interval == 0:
            count = step + 1
            cur_loss = round(total_loss / count, 2) 
            cur_nll = round(total_nll / count, 2) 
            cur_kl_theta = round(total_kl_theta_loss / count, 2) 
            cur_kl_eta = round(total_kl_eta_loss / count, 2) 
            cur_kl_alpha = round(total_kl_alpha_loss / count, 2) 
            lr = optimizer.param_groups[0]['lr']
            logging.debug('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, count, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))

        # log the losses to tensorboard
        writer.add_scalar('Loss/Total', loss.item(), epoch * len(indices) + step)
        writer.add_scalar('Loss/Cross-Entropy', nll.item(), epoch * len(indices) + step)
        writer.add_scalar('Loss/KL-Theta', kl_theta.item(), epoch * len(indices) + step)
        writer.add_scalar('Loss/KL-Eta', kl_theta.item(), epoch * len(indices) + step)
        writer.add_scalar('Loss/KL-Alpha', kl_theta.item(), epoch * len(indices) + step)

    final_total_loss = round(total_loss / len(indices), 2) 
    final_total_nll = round(total_nll / len(indices), 2) 
    final_total_kl_theta = round(total_kl_theta_loss / len(indices), 2) 
    final_total_kl_eta = round(total_kl_eta_loss / len(indices), 2) 
    final_total_kl_alpha = round(total_kl_alpha_loss / len(indices), 2) 
    lr = optimizer.param_groups[0]['lr']
    logging.debug('*'*100)
    logging.debug('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, lr, final_total_kl_theta, final_total_kl_eta, final_total_kl_alpha, final_total_nll, final_total_loss))
    logging.debug('*'*100)

    # save the model and optimizer after each epoch
    if epoch % args.save_interval == 0:
        torch.save(model.state_dict(), f'./checkpoints/detm_epoch{epoch}.pt')
        torch.save(optimizer.state_dict(), f'./checkpoints/optimizer_epoch{epoch}.pt')

for i in range(args.epochs):
    train(model, ann_data, optimizer, rnn_input, times_sorted, i)
torch.save(model.state_dict(), f'./checkpoints/detm_epoch{args.epoch}_final.pt')
torch.save(optimizer.state_dict(), f'./checkpoints/optimizer_epoch{args.epoch}_final.pt')
