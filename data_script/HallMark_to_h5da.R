library(anndata)
library(msigdbi)
library(magrittr)

pathway_DIR = '/home/BCCRC.CA/yzhang/projects/scCLR/data/pathways/'
# read in pathway information (Hallmark from MsigDB)
pathway_list = read.gmt(paste0(pathway_DIR, 'h.all.v7.5.1.symbols.gmt'))

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
write.table(data.frame(Gene_set = rownames(geneset_binary),geneset_binary,check.names=FALSE),file="/home/BCCRC.CA/yzhang/projects/scCLR/data/pathways/Hallmark.csv",row.names=FALSE,col.names=TRUE,quote=TRUE,sep=",")

ad <- AnnData(X = geneset_binary)
#ad$write_h5ad(filename = "/home/BCCRC.CA/yzhang/projects/scCLR/data/pathways/Hallmark.h5ad")
