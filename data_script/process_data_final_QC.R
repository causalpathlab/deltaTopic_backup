library(data.table)
library(Matrix)
library(stringr)
library(EnsDb.Hsapiens.v79)
library(msigdbi)
library(anndata)

data_DIR = paste0(getwd(),'/data/PDAC/pdac_velocity_topic/Results/QC/')
pathway_DIR = paste0(getwd(),'scCLR/data/pathways/')
# read in the data
counts = readMM(paste0(data_DIR, 'final_qc.mtx.gz'))
cols = fread(paste0(data_DIR, 'final_qc.cols.gz'), header=F)
rows = fread(paste0(data_DIR, 'final_qc.rows.gz'), header=F)
# read in pathway information (Hallmark from MsigDB)
pathway_list = read.gmt(paste0(pathway_DIR, 'h.all.v7.5.1.symbols.gmt'))
# read in meta data
meta_data = fread(paste0(data_DIR, 'Tab_Samples.csv'), header=T)
# create cancer type column
meta_data$Tissue_type = ifelse(meta_data$`Sample name` %>% str_detect(, pattern= 'N'), 'Normal', 'Tumor')

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

# create mapping betwwen ensembl id and gene symbol
ensembl.genes = gsub("\\..*","",rows$V1)
geneID <- ensembldb::select(EnsDb.Hsapiens.v79, keys= ensembl.genes, keytype = "GENEID", columns = c("SYMBOL","GENEID"))

# combine cols and meta data
samples_df = str_split(cols$V1, patter= "_", n = 2, simplify = TRUE) %>% as.data.frame()
names(samples_df) <- c('barcode', 'Accession')
samples_df <- dplyr::left_join(samples_df, meta_data, by = 'Accession')

# get the spliced genes
a1 = str_detect(rows$V1, "_spliced")
# check if the genes are in the pathway
geneID_in_pathway = geneID[geneID$SYMBOL %in% genes_in_all_genesets,]
a2 = str_detect(rows$V1, paste(geneID_in_pathway$GENEID, collapse="|"))

# create count matrix and gene list
counts_select = counts[a1 & a2,]

rows_select = rows$V1[a1 & a2] %>% str_split(pattern = "_", n = 2, simplify = TRUE) %>% as.data.frame()
names(rows_select) <- c('unique_gene_id', 'spliced')
#head(rows_select,5)
rows_select$GENEID <- gsub("\\..*","",rows_select$unique_gene_id)
#head(rows_select,5)
genes_df <- dplyr::left_join(rows_select, geneID,  by = 'GENEID')

ad <- AnnData(X = t(counts_select), 
              obs = data.frame(sample_id = samples_df$ID, tumor_type = samples_df$Tissue_type, sex = samples_df$Sex, row.names = cols$V1), 
              var = data.frame(gene = genes_df$SYMBOL, row.names = genes_df$SYMBOL))
ad$write_h5ad(filename = "/home/BCCRC.CA/yzhang/projects/scCLR/data/CRA001160/final_CRA001160_spliced.h5ad")
