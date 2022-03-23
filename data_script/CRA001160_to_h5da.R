library(anndata)
library(dplyr)
library(Matrix)
library(data.table)
library(tidyverse)
data_DIR = "data/CRA001160"
mtx = readMM(file = paste(data_DIR, "merged.mtx.gz", sep = "/"))
cell = fread(file = paste(data_DIR, "merged.columns.gz", sep = "/"))
features = fread(file = paste(data_DIR, "merged.rows.gz", sep = "/"), header = F)
# sample annotation with tumor type
sample = fread(file = paste(data_DIR, "samples.csv", sep = "/"),header = F, col.names = c("sample_id" , "tumor_type"))
# separate sample_id & join with tumor annotations
cell %>% separate(V1, c("barcode", "proj_id", "sample_id"), sep = "_") %>% left_join(sample) -> cell_new
rownames(cell_new) <- cell$V1

# save for PLIER
mtx_PLIER = t(mtx)
colnames(mtx_PLIER) <- features$V1
rownames(mtx_PLIER) <- cell$V1
save(mtx_PLIER, file = paste(data_DIR, "matrix_PLIER.RData", sep = "/"))
#######
ad <- AnnData(X = t(mtx), 
              obs = data.frame(sample_id = cell_new$sample_id, sample_idx = cell_new$V2, tumor_type = cell_new$tumor_type, row.names = rownames(cell_new)), 
              var = data.frame(gene = features$V1, row.names = features$V1))
ad$write_h5ad(filename = "CRA001160.h5ad")
