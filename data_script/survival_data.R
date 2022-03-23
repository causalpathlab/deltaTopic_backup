# impute survival information on GSE and CRA on the commoon genes with TCGA pancreas
# output a survival object for downstream analysis
library(data.table)
library(dplyr)
library(tidyr)
library(glmnet)
setwd("~/OneDrive - UBC/Github/scCLR/")

# If any of the processed files do not exit, then run the following
# to generate data files
if(!(file.exists("data/TCGA_pancreas/exp_subset.csv") & file.exists("data/TCGA_pancreas/exp_subset.csv"))){
  cat("Pre-proccessed data not found\nGenrating ...\n")
  exp_seq = fread(file = "data/TCGA_pancreas/exp_seq.PAAD-US.tsv.gz")
  donor = fread(file ="data/TCGA_pancreas/donor.PAAD-US.tsv.gz")
  
  seq_data = exp_seq %>% select(icgc_donor_id, analysis_id, gene_id, raw_read_count)
  # identify duplicate gene_id
  # seq_data %>% count(analysis_id,gene_id) %>% filter(n>1) %>% select(gene_id,n) %>% unique()
  #      ? 29
  #SLC35E2  2
  seq_data = seq_data[!(gene_id %in% c("SLC35E2","?")),] %>% pivot_wider(names_from = gene_id, values_from = raw_read_count) 
  
  # donor data
  donor$donor_survival_time[is.na(donor$donor_survival_time)] = 0
  donor$donor_interval_of_last_followup[is.na(donor$donor_interval_of_last_followup)] = 0
  donor$time = donor$donor_survival_time + donor$donor_interval_of_last_followup
  donor_subset = donor[time>0,.(icgc_donor_id,time,donor_vital_status,status = as.numeric(factor(donor_vital_status))-1)]
  
  fwrite(donor_subset, file =  "data/TCGA_pancreas/donor_subset.csv")
  fwrite(seq_data, file =  "data/TCGA_pancreas/exp_subset.csv")
}else{
  cat("Load pre-proccessed data from the folder\n")
  seq_data = fread("data/TCGA_pancreas/exp_subset.csv")
  donor_subset = fread("data/TCGA_pancreas/donor_subset.csv")
  } 

# survival analysis 
## get gene symbol form single cell RNA-seq
sc_data_path = "~/OneDrive - UBC/Github/scCLR/data/GSE154778/"
gene_sc = fread(file = paste0(sc_data_path,"merged.rows.gz"), header = F, col.names = "gene_symbol")
mtx = readMM(file = paste0(sc_data_path,"merged.mtx.gz"))

mtx = t(mtx)
cell = fread(file = paste0(sc_data_path,"merged.columns.gz"))
colnames(mtx) <- gene_sc$gene_symbol
rownames(mtx) <- cell$V1
# remove non-expressed genes
mtx = mtx[,apply(mtx, 2, sum)>0]

# match expression and sample meta 
merged_DT = merge(seq_data, donor_subset, by="icgc_donor_id") %>% as.data.table()
x = merged_DT[,-c("icgc_donor_id","analysis_id","time","donor_vital_status","status")] %>% as.matrix()

# remove non-expressed genes
x = x[,apply(x, 2, sum)>0]
# subset columns on the common genes
cm_genes = intersect(colnames(x), colnames(mtx))
x = x[,cm_genes] 
y = merged_DT[,c("time","status")] %>% as.matrix()
mtx = mtx[,cm_genes]

rowNorm=function (x) 
{
  s = apply(x, 1, sd)
  m = apply(x, 1, mean)
  x = sweep(x, 1, m)
  x = sweep(x, 1, s, "/")
  x
}
scaled = TRUE
if(scaled){
  rowNorm(t(x)) %>% t() -> x
  rowNorm(t(mtx)) %>% t() -> mtx
} 

set.seed(1)
fit <- glmnet(x, y, family = "cox")
cvfit <- cv.glmnet(x, y, family = "cox", type.measure = "C", nfolds = 10)
plot(fit)
plot(cvfit)

cvfit$lambda.min
cvfit$lambda.1se

# predict hazard on the single cell data

y_hat = predict(fit, newx = mtx, type = "response", s = cvfit$lambda.min)
plot(y_hat)


y2 = predict(fit, newx = x, type = "response", s = cvfit$lambda.min)
## # sample annotation with tumor type
sample = fread(file = paste0(sc_data_path,"samples.csv"),header = F, col.names = c("sample_id" , "tumor_type"))
# separate sample_id & join with tumor annotations
cell %>% separate(V1, c("barcode", "proj_id", "sample_id"), sep = "_") %>% left_join(sample) -> cell_new
rownames(cell_new) <- cell$V1

dim(cell_new)

df_to_plot = data.frame(y_hat = y_hat, tumor_type = cell_new$tumor_type, sample_id = cell_new$sample_id)
library(ggplot2)
ggplot(df_to_plot, aes(color = tumor_type))+geom_boxplot(x = log10(y_hat))

ggplot(df_to_plot,aes(sample_id, log10(y_hat), color = tumor_type, l))+geom_boxplot()+ylab('log10(y_hat)')


y_hat

cox_out = list(x_prime = mtx, y_hat = y_hat, y_hat_annot = cell_new, x_fit = x, y_fit = y, y_hat_in_sample = y2, fit = fit, cvfit = cvfit)
save(cox_out, file = "models/scCLR_tumor/cox_out.RData")

