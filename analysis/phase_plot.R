library(reticulate)
library(Matrix)
library(data.table)
library(ggplot2)
library(argparse)
library(magrittr)
library(ggpubr)

use_virtualenv('/home/BCCRC.CA/yzhang/DisNet',require = TRUE)

sc = import('scanpy')
os = import('os')
scipy = import('scipy')
parser <- ArgumentParser()

parser$add_argument("--SavePath",  
    help = "relative path to save folder")


args <- parser$parse_args()

#my_gene = 'NEAT1'
#'MALAT1'                              
 
SaveFolderPath = args$SavePath
#SaveFolderPath = "models/TotalDeltaETM_allgenes_ep2000_nlv16_bs512_combinebyadd_lr0.01_train_size1"
DataDIR = os$path$join(os$path$expanduser('~'), "projects/data")

adata_spliced = sc$read_h5ad(os$path$join(DataDIR,'CRA001160/final_CRA001160_spliced_allgenes.h5ad'))
adata_unspliced = sc$read_h5ad(os$path$join(DataDIR,'CRA001160/final_CRA001160_unspliced_allgenes.h5ad'))

sc$pp$normalize_per_cell(adata_spliced)
sc$pp$normalize_per_cell(adata_unspliced)
sc$pp$log1p(adata_spliced)
sc$pp$log1p(adata_unspliced)

S = Matrix(adata_spliced$X$toarray(), sparse=TRUE) 
colnames(S) <- adata_spliced$var$gene
rownames(S) <- adata_spliced$obs$sample_id 
U = Matrix(adata_unspliced$X$toarray(), sparse=TRUE) 
colnames(U) <- adata_unspliced$var$gene
rownames(U) <- adata_unspliced$obs$sample_id

rm(adata_spliced, adata_unspliced)

topics_df = fread(paste0(SaveFolderPath,'/topics.csv'))
rownames(topics_df) <-topics_df$V1
topics_df = topics_df[,-1]
topics_df$topics <- colnames(topics_df)[apply(topics_df,1,which.max)]

my_gene_list_df = read.csv(paste0(SaveFolderPath, paste0("/","delta","_topK_genes.csv")))

K = 3
my_gene_list = c()
for(j in 1:ncol(my_gene_list_df)){
    print(as.character(my_gene_list_df[1:K,j]))
    my_gene_list = c(my_gene_list, as.character(my_gene_list_df[1:K,j]))     
}
my_gene_list = as.factor(my_gene_list) %>% unique()

for(my_gene in my_gene_list){
    df = data.frame(S = S[,my_gene], U = U[,my_gene])
    df = cbind(df, topics_df)

    p1 <- ggplot(df, aes(x = S, y = U,color = topics)) + 
    geom_point(alpha = 0.1) + 
    facet_grid(~topics,scale = "fixed") + 
    #geom_density_2d() +
    geom_abline(intercept = 0, slope = 1, colour = "red") + 
    ggtitle(my_gene)
    #ggsave(paste0(SaveFolderPath,'/',my_gene,'_scatter_topics.png'))

    p2 <- ggplot(df, aes(x = S, y = U, color = topics)) +
    geom_point(alpha = 0.5) +
    #geom_jitter() + 
    geom_abline(intercept = 0, slope = 1, colour = "red") + 
    ggtitle(my_gene)
    ggarrange(p1, p2, 
          labels = c("A", "B"),
          ncol = 1, nrow = 2)
    ggsave(paste0(SaveFolderPath,'/',my_gene,'_scatter.png'))
}

