import os
import random
import pickle
import numpy as np
import torch 
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'tokens_1': tokens_1, 'counts_1': counts_1, 'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}

def _fetch_temporal(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
        time_file = os.path.join(path, 'bow_tr_timestamps.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
        time_file = os.path.join(path, 'bow_va_timestamps.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
        time_file = os.path.join(path, 'bow_ts_timestamps.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'times': times, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2} 
    return {'tokens': tokens, 'counts': counts, 'times': times}

def get_data(path, temporal=False):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train = _fetch(path, 'train')
        valid = _fetch(path, 'valid')
        test = _fetch(path, 'test')
    else:
        train = _fetch_temporal(path, 'train')
        valid = _fetch_temporal(path, 'valid')
        test = _fetch_temporal(path, 'test')

    return vocab, train, valid, test

def get_batch(tokens, counts, ind, vocab_size, emsize=300, temporal=False, times=None):
    """fetch input data by batch."""
    # batch size is based on the batch itself
    batch_size = len(ind)
    # 1000 samples x genes
    data_batch = np.zeros((batch_size, vocab_size))

    # if it's temporal (which we care about) we'll maintain a vector of 1000 samples
    if temporal:
        times_batch = np.zeros((batch_size, ))

    # then for each sample in the 1000 samples
    for i, doc_id in enumerate(ind):
        # get the specific cell
        doc = tokens[doc_id]
        # then get the gene count per document
        count = counts[doc_id]
        if temporal:
            timestamp = times[doc_id]
            times_batch[i] = timestamp
        L = count.shape[1]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, times_batch
    return data_batch

# formats the rnn inputs -- basically returns a time x vocab list
# of how many times a token was seen per time step
def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs):
    # first, get a random permutation over all the cells
    indices = torch.randperm(num_docs)
    # then, we split this up into groups of 1000
    indices = torch.split(indices, 1000)

    # then, we create a times by genes data count
    rnn_input = torch.zeros(num_times, vocab_size).to(device)

    # cnt provides a count for how many documents per time step we encounter
    cnt = torch.zeros(num_times, ).to(device)

    # iterate over all the random permutations of cells
    for idx, ind in enumerate(indices):
        # we grab the batch:
            # data_batch represents
            # times_batch represents
        data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)

        # then for each time step
        for t in range(num_times):
            # we check if the times match up
            # tmp is the list of indices that are non-zero
            tmp = (times_batch == t).nonzero()
            # then for each of these indices, we add up the count I think

            # ! Note: there is a slight bug in the code here when we don't have
            # ! more than 1 sample per time step
            docs = data_batch[tmp].squeeze().sum(0)
            # then, we add this to the time step's gene count I think...
            rnn_input[t] += docs
            # add to the total count for that time step
            cnt[t] += len(tmp)

        if idx % 20 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))
    
    # then normalize it by the total counts for each time step
    rnn_input = rnn_input / cnt.unsqueeze(1)

    # return the times by genes matrix that we have.
    return rnn_input
