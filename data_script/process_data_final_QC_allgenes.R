library(data.table)
library(Matrix)
library(stringr)
library(EnsDb.Hsapiens.v86)
library(msigdbi)
library(anndata)

data_DIR = '/home/yichenzhang/projects/data/CRA001160/PDAC/pdac_velocity_topic/Results/QC/'
pathway_DIR = '/home/yichenzhang/projects/data/pathways/'

paste(save_file_DIR, paste0("final_CRA001160_", splicing,"_allgenes.h5ad"), sep = '')
# read in the data
counts = readMM(paste0(data_DIR, 'final_qc.mtx.gz'))
dim(counts)
cols = fread(paste0(data_DIR, 'final_qc.cols.gz'), header=F)
dim(cols)
rows = fread(paste0(data_DIR, 'final_qc.rows.gz'), header=F)
dim(rows)

# read in meta data
meta_data = fread(paste0(data_DIR, 'Tab_Samples.csv'), header=T)
# create cancer type column
meta_data$Tissue_type = ifelse(meta_data$`Sample name` %>% str_detect(, pattern= 'N'), 'Normal', 'Tumor')
dim(meta_data)

'''# read in pathway information (Hallmark from MsigDB)
# pathway_list = read.gmt(paste0(pathway_DIR, 'h.all.v7.5.1.symbols.gmt'))
# ceate a binary matrix for pathway
geneset_list_to_df <- function(vects) { 
    lev <- sort(unique(unlist(vects)))
    dat <- do.call(rbind, lapply(vects, function(x, lev){ 
        tabulate(factor(x, levels = lev, ordered = TRUE),
        nbins = length(lev))}, lev = lev))
    colnames(dat) <- sort(lev) 
    data.frame(dat, check.names = FALSE)
}
geneset_binary <- geneset_list_to_df(pathway_list$genesets)
# Output this binary matrix to a CSV file.
# write.table(data.frame(Gene_set = rownames(binary_matrix),binary_matrix,check.names=FALSE),file="GO_MF_binary_matrix.csv",row.names=FALSE,col.names=TRUE,quote=TRUE,sep=",")
# get all gene symbols in the pathway
genes_in_all_genesets <- geneset_binary %>% colnames()
'''

# create mapping betwwen ensembl id and gene symbol
ensembl.genes = gsub("\\..*","",rows$V1)
length(ensembl.genes)
geneID <- ensembldb::select(EnsDb.Hsapiens.v86, keys= ensembl.genes, keytype = "GENEID", columns = c("SYMBOL","GENEID"))


# combine cols and meta data
samples_df = str_split(cols$V1, patter= "_", n = 2, simplify = TRUE) %>% as.data.frame()
names(samples_df) <- c('barcode', 'Accession')
samples_df <- dplyr::left_join(samples_df, meta_data, by = 'Accession')

###
splicing = "unspliced"
# get the spliced genes
a1 = str_detect(rows$V1, paste0("_",splicing))
# check if the genes are in the pathway
#geneID_in_pathway = geneID[geneID$SYMBOL %in% genes_in_all_genesets,]
#a2 = str_detect(rows$V1, paste(geneID_in_pathway$GENEID, collapse="|"))

# create count matrix and gene list
counts_select = counts[a1,]
#counts_select = counts[a1 & a2,]

rows_select = rows$V1[a1] %>% str_split(pattern = "_", n = 2, simplify = TRUE) %>% as.data.frame()
names(rows_select) <- c('unique_gene_id', splicing)
head(rows_select,5)
rows_select$GENEID <- gsub("\\..*","",rows_select$unique_gene_id)
head(rows_select,5)
genes_df <- dplyr::left_join(rows_select, geneID,  by = 'GENEID')
dim(genes_df) 
dim(counts_select)
# remove the no-match genes
no_match_rows = genes_df$SYMBOL %>% is.na()
genes_df = genes_df[!no_match_rows,] 
counts_select = counts_select[!no_match_rows,]
dim(genes_df) 
dim(counts_select)
# check to see if there are duplicates
if(sum(genes_df$SYMBOL %>% duplicated()) > 0){
    print("only keep the first appearance when duplicates exist")
    duplicated_rows = genes_df$SYMBOL %>% duplicated()
    genes_df = genes_df[!duplicated_rows,]
    counts_select = counts_select[!duplicated_rows,]
    dim(genes_df) 
    dim(counts_select)
}

ad <- AnnData(X = t(counts_select), 
              obs = data.frame(sample_id = samples_df$ID, tumor_type = samples_df$Tissue_type, sex = samples_df$Sex, row.names = cols$V1), 
              var = data.frame(unique_gene_id = genes_df$unique_gene_id, gene = genes_df$SYMBOL, row.names = genes_df$SYMBOL))
ad$write_h5ad(filename = paste("/home/yichenzhang/projects/data/CRA001160/final_CRA001160",splicing, "allgenes.h5ad", sep = "_"))
