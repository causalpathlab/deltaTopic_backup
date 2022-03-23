#%% load the trained model
import pandas as pd
from DeltaETM_model import DeltaETM
import scanpy as sc
import matplotlib.pyplot as plt
model = DeltaETM.load("models/DeltaETM_allgenes_ep50_nlv4_bs256")
#%% get LVs

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

