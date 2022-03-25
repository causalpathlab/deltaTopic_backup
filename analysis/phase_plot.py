import matplotlib.pyplot as plt
import os
import scanpy as sc
import pandas as pd



SaveFolderPath = 'models/TotalDeltaETM_allgenes_ep2000_nlv4_bs512_combinebyadd'
DataDIR = os.path.join(os.path.expanduser('~'), "projects/data")

adata_spliced = sc.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_spliced_allgenes.h5ad'))
adata_unspliced = sc.read_h5ad(os.path.join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))

sc.pp.normalize_per_cell(adata_spliced)
sc.pp.normalize_per_cell(adata_unspliced)
sc.pp.log1p(adata_spliced)
sc.pp.log1p(adata_unspliced)

topics_df = pd.read_csv(os.path.join(SaveFolderPath,'topics.csv'),index_col=0)
topics_df["topics"] = topics_df.idxmax(1)
colors = {'topic_0':'red', 'topic_1':'green', 'topic_2':'blue', 'topic_3':'yellow'}

my_genes_list = ['MALAT1','B2M','CYBA','SLC4A4',"GAS5"]

for my_gene in my_genes_list:
    if os.path.exists(os.path.join(SaveFolderPath,f"phaseplot_{my_gene}.png")):
        continue
    S = adata_spliced[:,my_gene].X.toarray()
    U = adata_unspliced[:,my_gene].X.toarray()    
    fig = plt.figure()
    plt.scatter(x = U, y = S, alpha=0.5, c=topics_df["topics"].map(colors))
    plt.title(my_gene)
    plt.xlabel("Unspliced")
    plt.ylabel("Spliced")
    plt.savefig(os.path.join(SaveFolderPath,f"phaseplot_{my_gene}.png"))


