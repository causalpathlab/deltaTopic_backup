import anndata
import os
import scvi
from scipy.sparse import csr_matrix
#from scipy.stats import spearmanr
from scvi.data import setup_anndata
import wandb
import argparse
from scipy.sparse import csr_matrix
import scanpy
#%%
wandb.login()
wandb.init(entity="thisisyichen")
# Input parser
parser = argparse.ArgumentParser(description='Parameters for NN')
parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=500)
#parser.add_argument('--lr', type=float, help='learning_rate', default=1e-3)
#parser.add_argument('--CUDA', type=int, help='which GPU to use', default=0)
parser.add_argument('--nLV', type=int, help='User specified nLV', default=4)
parser.add_argument('--bs', type=int, help='Batch size', default=256)
#parser.add_argument('--concat_method', type=str, help='Pathway type', default='cat')
args = parser.parse_args()
# pass args to wand.config
wandb.config.update(args)
#%%
DataDIR = os.path.join(os.path.expanduser('~'), "projects/data")
adata_spliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_spliced_allgenes.h5ad'))
adata_unspliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))
# for speed-up of training
adata_spliced.layers["counts"] = csr_matrix(adata_spliced.X).copy()
adata_unspliced.layers["counts"] = csr_matrix(adata_unspliced.X).copy()
# save raw adata 
adata_spliced.raw = adata_spliced
adata_unspliced.raw = adata_unspliced
#%%
'''pathways = anndata.read_h5ad(os.path.join(DataDIR,'pathways/Hallmark.h5ad'))
# concat relevant pathways

#ad_tst = anndata.read_h5ad("data/pathways/chemgenPathways.h5ad")
#ad_tst2 = anndata.read_h5ad("data/pathways/immunePathways.h5ad")
#ad_tst3 = anndata.read_h5ad("data/pathways/bloodCellMarkersIRISDMAP.h5ad")
#ad_tst4 = anndata.read_h5ad("data/pathways/canonicalPathways.h5ad")
#ad_tst5 = anndata.read_h5ad("data/pathways/oncogenicPathways.h5ad")
#ad_tst6 = anndata.read_h5ad("data/pathways/xCell.h5ad")
#pathways = anndata.concat([ad_tst, ad_tst2,ad_tst3, ad_tst4,ad_tst5, ad_tst6], join="outer", fill_value = 0)'''
#%%
setup_anndata(adata_spliced, layer="counts", batch_key="sample_id")
setup_anndata(adata_unspliced, layer="counts", batch_key="sample_id")
#%% 
# set up anndata
# get the common genes in pathway and adata
'''df_cm = adata_spliced.var.join(pathways.var, how = 'inner')
# register adata_spliced
adata_spliced_input = adata_spliced[:,df_cm.index.to_list()].copy()
setup_anndata(adata_spliced_input, layer="counts", batch_key="sample_id")

# register adata_unspliced
adata_unspliced_input = adata_unspliced[:,df_cm.index.to_list()].copy()
scanpy.pp.filter_cells(adata_unspliced_input, min_counts=10)
setup_anndata(adata_unspliced_input, layer="counts", batch_key="sample_id")
# register pathway 
pathways.var_names_make_unique()
pathways_input = pathways[:,df_cm.index.to_list()].copy()
setup_anndata(pathways_input)'''
#%%
#create our model
from DeltaETM_model import DeltaETM
model = DeltaETM(adata_spliced, adata_unspliced, n_latent = args.nLV)
#model = DeltaETM(adata_spliced_input, adata_unspliced_input, adata_pathway = pathways_input)
#model = scCLR(adata_tumor_input, adata_metastatic_input, mask = torch.bernoulli(torch.empty(100, 16445).uniform_(0, 1)))
#%%
from pytorch_lightning.loggers import WandbLogger
# this has to be passed, otherwise pytroch lighting logging won't be passed to wandb
wandb_logger = WandbLogger(project = 'DeltaETM')
#model_kwargs = {"lr": args.lr}
model.train(
    args.EPOCHS, 
    check_val_every_n_epoch=5,
    batch_size=args.bs,
    logger = wandb_logger,
#    model_kwargs,
    )
#%%
savefile_name = f"models/DeltaETM_allgenes_ep{args.EPOCHS}_nlv{args.nLV}_bs{args.bs}"
model.save(savefile_name, overwrite=True, save_anndata=True)
print(f"Model saved to {savefile_name}")
########


'''import scanpy as sc
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