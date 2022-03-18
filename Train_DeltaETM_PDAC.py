import torch
import scanpy
import anndata
import numpy as np
import copy
import os
#import matplotlib.pyplot as plt
import scvi
from scipy.sparse import csr_matrix
#from scipy.stats import spearmanr
from scvi.data import setup_anndata
import wandb
import argparse
from scipy.sparse import csr_matrix
#%%
wandb.login()
#wandb.init(entity="thisisyichen")
# Input parser
#parser = argparse.ArgumentParser(description='Parameters for NN')
#parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=500)
#parser.add_argument('--learning_rate', type=float, help='learning_rate', default=1e-3)
#parser.add_argument('--nLV', type=int, help='User specified nLV', default=10)
#parser.add_argument('--concat_method', type=str, help='Pathway type', default='cat')
#args = parser.parse_args()
# pass args to wand.config
#wandb.config.update(args)
#%%
homeDIR = os.path.join(os.path.expanduser('~'))
os.path.join(homeDIR,'projects/scCLR/data/CRA001160/final_CRA001160_spliced.h5ad')
adata = scvi.data.read_h5ad(os.path.join(homeDIR,'projects/scCLR/data/CRA001160/final_CRA001160_spliced.h5ad'))
# for speed-up of training
adata.layers["counts"] = csr_matrix(adata.X).copy()
# save raw adata 
adata.raw = adata
# get PDAC and normal samples
adata_PDAC = adata[adata.obs.tumor_type == "Tumor",:]
adata_normal = adata[adata.obs.tumor_type == "Normal",:]
# concat relevant pathways
pathways = anndata.read_h5ad(os.path.join(homeDIR,'projects/scCLR/data/pathways/Hallmark.h5ad'))
#ad_tst = anndata.read_h5ad("data/pathways/chemgenPathways.h5ad")
#ad_tst2 = anndata.read_h5ad("data/pathways/immunePathways.h5ad")
#ad_tst3 = anndata.read_h5ad("data/pathways/bloodCellMarkersIRISDMAP.h5ad")
#ad_tst4 = anndata.read_h5ad("data/pathways/canonicalPathways.h5ad")
#ad_tst5 = anndata.read_h5ad("data/pathways/oncogenicPathways.h5ad")
#ad_tst6 = anndata.read_h5ad("data/pathways/xCell.h5ad")
#pathways = anndata.concat([ad_tst, ad_tst2,ad_tst3, ad_tst4,ad_tst5, ad_tst6], join="outer", fill_value = 0)
#%% 
# set up anndata
# get the common genes in pathway and adata
df_cm = adata_PDAC.var.join(pathways.var, how = 'inner')
# register adata_source1
adata_PDAC_input = adata_PDAC[:,df_cm.index.to_list()].copy()
setup_anndata(adata_PDAC_input, layer="counts", batch_key="sample_id", labels_key="tumor_type")

# register adata_source2
adata_normal_input = adata_normal[:,df_cm.index.to_list()].copy()
setup_anndata(adata_normal_input, layer="counts", batch_key="sample_id", labels_key="tumor_type")
# register pathway 
pathways.var_names_make_unique()
pathways_input = pathways[:,df_cm.index.to_list()].copy()
setup_anndata(pathways_input)
#%%
#create our model
from DeltaETM_model import DeltaETM
import torch
model = DeltaETM(adata_PDAC_input, adata_normal_input, adata_pathway = pathways_input)
#model = scCLR(adata_tumor_input, adata_metastatic_input, mask = torch.bernoulli(torch.empty(100, 16445).uniform_(0, 1)))
#%%
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project = 'DeltaETM')
model.train(
    5, 
    check_val_every_n_epoch=5,
    batch_size=256,
    logger=wandb_logger, kappa=1000)


'''model.save("models/scCLR_masked_decoder_pancreas_final_QC_no_softmax_new", overwrite=True, save_anndata=True)

########


import scanpy as sc
import matplotlib.pyplot as plt

# concatenate adata to get all cells
adatas = adata_PDAC_input.concatenate(adata_normal_input)
# load trained model
model.load("models/scCLR_masked_decoder_pancreas_final_QC_no_softmax_new")
# get latent dimensions
cancer_latents, normal_latents = model.get_latent_representation()
normal_on_cancer_LV, cancer_on_normal_LV = model.get_latent_representation(adata_normal_input,adata_PDAC_input)

Z_shared = np.concatenate([cancer_latents['latent_z'], normal_latents['latent_z']])
Z_shared_df = pd.DataFrame(Z_shared, index = adatas.obs.index)
adatas.obsm["Z_shared"] = Z_shared_df
sc.pp.neighbors(adatas, use_rep="Z_shared", n_neighbors=20)
sc.tl.umap(adatas, min_dist=0.3)
sc.tl.leiden(adatas, key_added="leiden_scVI", resolution=0.8)
# color by tumor type
fig = plt.figure()
sc.pl.umap(adatas, color=["tumor_type"])
plt.savefig('UMAP_all_zshared_by_tumor.png', bbox_inches='tight')


# print out the pathway loadings
loadings_cancer, loadings_normal, loadings_shared = model.get_loadings()
loadings_cancer.head()
loadings = loadings_cancer
print('Top loadings by magnitude\n---------------------------------------------------------------------------------------')
for clmn_ in loadings:
    loading_ = loadings[clmn_].sort_values()
    fstr = clmn_ + ':\t'
    fstr += '\t'.join([f'{i}, {loading_[i]:.2}' for i in loading_.head(5).index])
    fstr += '\n\t...\n\t'
    fstr += '\t'.join([f'{i}, {loading_[i]:.2}' for i in loading_.tail(5).index])
    print(fstr + '\n---------------------------------------------------------------------------------------\n')


zs = [f'Z1_{i}' for i in range(model.n_latent)]
fig = plt.figure()
sc.pl.umap(adatas, color=zs, ncols=3)
plt.savefig('UMAP_all_zshared_by_z1.png', bbox_inches='tight')

import scanpy as sc
import matplotlib.pyplot as plt
sc.pp.neighbors(adatas, n_neighbors=20)
sc.tl.umap(adatas, min_dist=0.3)
#sc.tl.leiden(adatas, key_added="leiden_scVI", resolution=0.8)
fig = plt.figure()
#adatas.obsm['tumor_type'] = adatas.obs.tumor_type
sc.pl.umap(adatas, color = ["tumor_type"])

plt.savefig('UMAP_combined_raw_gene.png')'''