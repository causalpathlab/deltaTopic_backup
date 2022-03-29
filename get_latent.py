#%% load the trained model
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import argparse
#%% get LVs for TotalDeltaETM
from DeltaETM_model import TotalDeltaETM

plot_gene_space = False
'''
parser = argparse.ArgumentParser(description='Parameters for NN')

parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=2000)
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-2)
parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
parser.add_argument('--nLV', type=int, help='User specified nLV', default=16)
parser.add_argument('--bs', type=int, help='Batch size', default=512)
parser.add_argument('--combine_method', type=str, help='Pathway type', default='add')
parser.add_argument('--train_size', type=float, help='training size', default=1)

args = parser.parse_args()'''
parser = argparse.ArgumentParser(description='Parameters for NN')

parser.add_argument('--SavePath', type=str, help='path to save')
args = parser.parse_args()
SaveFolderPath = args.SavePath
#SaveFolderPath = f"models/TotalDeltaETM_allgenes_ep{args.EPOCHS}_nlv{args.nLV}_bs{args.bs}_combineby{args.combine_method}_lr{args.lr}_train_size{args.train_size}"
print(SaveFolderPath)


model = TotalDeltaETM.load(SaveFolderPath)

topics_np = model.get_latent_representation()
topics_untran_np = model.get_latent_representation(output_softmax_z=False)


topics_df = pd.DataFrame(topics_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
topics_df.to_csv(os.path.join(SaveFolderPath,"topics.csv"))
topics_untran_df = pd.DataFrame(topics_untran_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_untran_np.shape[1])])
topics_untran_df.to_csv(os.path.join(SaveFolderPath,"topics_untran.csv"))

# get the weight matrix

delta, rho, log_delta, log_rho = model.get_weights()
rho_df = pd.DataFrame(rho, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
rho_df.to_csv(os.path.join(SaveFolderPath,"rho_weights.csv"))
delta_df = pd.DataFrame(delta, index = ['topic_' + str(j) for j in range(topics_np.shape[1])], columns = model.adata.var.index).T
delta_df.to_csv(os.path.join(SaveFolderPath,"delta_weights.csv"))

#%%
model.adata.obsm['X_DeltaETM_topic'] = topics_df
model.adata.obsm["X_DeltaETM_topic_untran"] = topics_untran_df
for i in range(topics_np.shape[1]):
    model.adata.obs[f"DeltaETM_topic_{i}"] = topics_df[[f"topic_{i}"]]
    model.adata.obs[f"DeltaETM_topic_untran_{i}"] = topics_untran_df[[f"topic_{i}"]]
#%% plot UMAP on topic space
model.adata.obs['sample_id_cat'] = model.adata.obs['sample_id'].astype('category',copy=False)

sc.pp.neighbors(model.adata, use_rep="X_DeltaETM_topic")
sc.tl.umap(model.adata)
# Save UMAP to custom .obsm field.
model.adata.obsm["topic_space_umap"] = model.adata.obsm["X_umap"].copy()
fig = plt.figure()
sc.pl.embedding(model.adata, "topic_space_umap", color = [f"DeltaETM_topic_{i}" for i in range(topics_np.shape[1])], frameon=False)
plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic.png'))

sc.pl.embedding(model.adata, "topic_space_umap", color = ['tumor_type','sample_id_cat'], frameon=False)
plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic_by_tumor_sample.png'),bbox_inches='tight')

#%% plot UMAP on topic space untransformed
model.adata.obs['sample_id_cat'] = model.adata.obs['sample_id'].astype('category',copy=False)
sc.pp.neighbors(model.adata, use_rep="X_DeltaETM_topic_untran")
sc.tl.umap(model.adata)
# Save UMAP to custom .obsm field.
model.adata.obsm["topic_space_umap_untran"] = model.adata.obsm["X_umap"].copy()
fig = plt.figure()
sc.pl.embedding(model.adata, "topic_space_umap_untran", color = [f"DeltaETM_topic_untran_{i}" for i in range(topics_np.shape[1])], frameon=False)
plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic_untran.png'))

sc.pl.embedding(model.adata, "topic_space_umap_untran", color = ['tumor_type','sample_id_cat'], frameon=False)
plt.savefig(os.path.join(SaveFolderPath,'UMAP_topic_by_tumor_sample_untran.png'),bbox_inches='tight')




if plot_gene_space:
    
    # plot UMAP on spliced count space 
    sc.pp.neighbors(model.adata, n_pcs = 10, use_rep="X")
    sc.tl.umap(model.adata)
    # Save UMAP to custom .obsm field.
    model.adata.obsm["spliced_umap"] = model.adata.obsm["X_umap"].copy()
    fig = plt.figure()
    sc.pl.embedding(model.adata, "spliced_umap", color = [f"DeltaETM_topic_{i}" for i in range(topics_np.shape[1])], frameon=False)
    plt.savefig(os.path.join(SaveFolderPath,'UMAP_spliced.png'))
    fig = plt.figure()
    sc.pl.embedding(model.adata, "spliced_umap", color = ['tumor_type','sample_id_cat'], frameon=False)
    plt.savefig(os.path.join(SaveFolderPath,'UMAP_spliced_by_tumor_sample.png'),bbox_inches='tight')
    
    # plot UMAP on unspliced count space
    sc.pp.neighbors(model.adata, n_pcs = 10, use_rep="protein_expression")
    sc.tl.umap(model.adata)
    # Save UMAP to custom .obsm field.
    model.adata.obsm["unspliced_umap"] = model.adata.obsm["X_umap"].copy()
    fig = plt.figure()
    sc.pl.embedding(model.adata, "unspliced_umap", color = [f"DeltaETM_topic_{i}" for i in range(topics_np.shape[1])], frameon=False)
    plt.savefig(os.path.join(SaveFolderPath,'UMAP_unspliced.png')) 
    fig = plt.figure()
    sc.pl.embedding(model.adata, "unspliced_umap", color = ['tumor_type','sample_id_cat'], frameon=False)
    plt.savefig(os.path.join(SaveFolderPath,'UMAP_unspliced_by_tumor_sample.png'),bbox_inches='tight')

'''#%% get LVs for DeltaETM
# This is for DeltaETM model where 'z' is saved as dict
# For TotalETM model, 'z' is saved as numpy array, refer to the other function
from DeltaETM_model import DeltaETM
model = DeltaETM.load("models/DeltaETM_allgenes_ep50_nlv4_bs256")
topics_list = model.get_latent_representation()
topics_np = []
topics_df = []

for i in range(2):
    topics_np.append(topics_list[i]['latent_z'])

for i in range(2):
    topics_df.append(pd.DataFrame(topics_np[i], index= model.adatas[i].obs.index, columns = ['topic_' + str(j) for j in range(4)]))
    model.adatas[i].obsm['X_DeltaETM'] = topics_df[i]

for j in range(2):
    for i in range(4):
        model.adatas[j].obs[f"DeltaETM_topic_{i}"] = topics_df[j][[f"topic_{i}"]]

#%% plot UMAP in topic space
for i in range(i):   
    sc.pp.neighbors(model.adatas[i], use_rep="X_DeltaETM")
    sc.tl.umap(model.adatas[i])
    # Save UMAP to custom .obsm field.
    model.adatas[i].obsm["topic_space_umap"] = model.adatas[i].obsm["X_umap"].copy()
    fig = plt.figure()
    sc.pl.embedding(model.adatas[i], "topic_space_umap", color = [f"DeltaETM_topic_{i}" for i in range(4)], frameon=False)
    plt.savefig(f'UMAP_topic_space_adata{i}.png')

#%% plot UMAP
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_pcs = 30, n_neighbors = 20)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added = "leiden_scVI", resolution = 0.8)
'''
