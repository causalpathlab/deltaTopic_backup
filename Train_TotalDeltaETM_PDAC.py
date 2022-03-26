import os
import scvi
from scipy.sparse import csr_matrix
from scvi.data import setup_anndata

import argparse
from scipy.sparse import csr_matrix
#%%
import wandb
wandb.login()
wandb.init(entity="thisisyichen", project="TotalDeltaETM")
# Input parser
parser = argparse.ArgumentParser(description='Parameters for NN')
parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=2000)
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-2)
parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
parser.add_argument('--nLV', type=int, help='User specified nLV', default=4)
parser.add_argument('--bs', type=int, help='Batch size', default=512)
parser.add_argument('--combine_method', type=str, help='Pathway type', default='add')
parser.add_argument('--train_size', type=float, help='training size', default=1)
args = parser.parse_args()
# pass args to wand.config
wandb.config.update(args)
#%%
savefile_name = f"models/TotalDeltaETM_allgenes_ep{args.EPOCHS}_nlv{args.nLV}_bs{args.bs}_combineby{args.combine_method}_lr{args.lr}_train_size{args.train_size}"
print(savefile_name)
DataDIR = os.path.join(os.path.expanduser('~'), "projects/data")
adata_spliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_spliced_allgenes.h5ad'))
adata_unspliced = scvi.data.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))


# for speed-up of training
adata_spliced.layers["counts"] = csr_matrix(adata_spliced.X).copy()
adata_spliced.obsm["protein_expression"] = csr_matrix(adata_unspliced.X).copy()
setup_anndata(adata_spliced, layer="counts", batch_key="sample_id", protein_expression_obsm_key = "protein_expression")

#adata_unspliced.layers["counts"] = csr_matrix(adata_unspliced.X).copy()
# save raw adata 
#adata_spliced.raw = adata_spliced
#adata_unspliced.raw = adata_unspliced
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
#
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
from DeltaETM_model import TotalDeltaETM
#model = TotalDeltaETM(adata_spliced)
#model = TotalDeltaETM.load("models/TotalDeltaETM_allgenes_ep2_nlv4_bs256")
#lv = model.get_latent_representation()
model = TotalDeltaETM(adata_spliced, n_latent = args.nLV, combine_latent= args.combine_method)
#model = DeltaETM(adata_spliced_input, adata_unspliced_input, adata_pathway = pathways_input)
#model = scCLR(adata_tumor_input, adata_metastatic_input, mask = torch.bernoulli(torch.empty(100, 16445).uniform_(0, 1)))
#%%
# this has to be passed, otherwise pytroch lighting logging won't be passed to wandb
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project = 'TotalDeltaETM')
model_kwargs = {"lr": args.lr, 'use_gpu':args.use_gpu, 'train_size':args.train_size}

print(args)
model.train(
    args.EPOCHS, 
    #check_val_every_n_epoch=5,
    batch_size=args.bs,
    logger = wandb_logger,
    **model_kwargs,
    )
#%%
model.save(savefile_name, overwrite=True, save_anndata=True)
print(f"Model saved to {savefile_name}")
########
