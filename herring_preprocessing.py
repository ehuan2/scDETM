import scanpy as sc
import anndata

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

print(f'Male counts: {ann_data_male.shape}, Female counts: {ann_data_female}')
