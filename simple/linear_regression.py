import scanpy as sc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from bulk_analyze import get_bulk_data

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default='../data/Processed_data_RNA-gaba_full-counts-and-downsampled-CPM.h5ad')
args = argparser.parse_args()

scrna_seq = sc.read_h5ad(args.data_path)

bulk_data, times_sorted, idx_and_cell_types = get_bulk_data(scrna_seq)