library(ggplot2)
library(data.table)
library(gplots)
library(magrittr)
library(argparse)
parser <- ArgumentParser()

parser$add_argument("--SavePath",  
    help = "relative path to save folder")
parser$add_argument("--target", default= "delta", 
    help = "weight to plot, delta or rho")

args <- parser$parse_args()    
Save_Path = args$SavePath
target <- args$target
#target = "delta"
fileDIR = paste0(Save_Path, "/")
#"models/TotalDeltaETM_allgenes_ep2000_nlv16_bs512_combinebyadd_lr0.01_train_size1/"

weights_dt = fread(paste0(fileDIR, paste0(target,"_weights.csv")))
weights = weights_dt %>% as.data.frame()
#pdf(file = paste0(fileDIR, paste0(target,"_weights.pdf")))
weight_mat <- weights %>% as.data.frame()
rownames(weight_mat) = weights$V1
weight_mat = as.matrix(weight_mat[,-1])
#weight_mat %>%  gplots::heatmap.2()
#dev.off()


ngenes = nrow(weight_mat)

ordered_weights_matrix = matrix(nrow = ngenes, ncol = ncol(weight_mat))
ordered_genes_matrix = matrix(nrow = ngenes, ncol = ncol(weight_mat))


for(i in 1:ncol(weight_mat)){
    print(i)
    weights_order = weights[weights[,paste0('topic_',i-1)] %>% order(decreasing = T),paste0('topic_',i-1)]
    genes_order = weights[weights[,paste0('topic_',i-1)] %>% order(decreasing = T),'V1']
    position = c(1:ngenes)
    ordered_weights_matrix[,i] <- weights_order[position]
    ordered_genes_matrix[,i] <- genes_order[position]

}

colnames(ordered_weights_matrix) = colnames(weight_mat)
colnames(ordered_genes_matrix) = colnames(weight_mat)

library(topicmodels)
library(wordcloud)
K = 20 # top K genes to plot

#topic = 1
for(topic in 1:ncol(ordered_weights_matrix)){
    probabilities = ordered_weights_matrix[1:K,topic]
    genes = ordered_genes_matrix[1:K,topic]

    mycolors <- brewer.pal(8, "Dark2")
    png(file = paste0(fileDIR, paste0("gene_cloud_top",K,"_",colnames(ordered_weights_matrix)[topic],".png")))
    wordcloud(genes, probabilities, 
        random.order = FALSE, color = mycolors,
        main = colnames(ordered_weights_matrix)[topic])
    dev.off()
}


#data.table::fwrite(as.data.frame(ordered_genes_matrix), file = paste0(fileDIR, paste0(target,"_topK_genes.csv")))
#data.table::fwrite(as.data.frame(ordered_weights_matrix), file = paste0(fileDIR, paste0(target,"_topK_weights.csv")))


K = 10 # top K genes and bottom K for each topic

weight_mat_topK = ordered_weights_matrix[1:K,]
gene_mat_topK = ordered_genes_matrix[1:K,]
pdf(file = paste0(fileDIR, paste0("Heatmap_top_genes_",target,"_weights.pdf")))
#myColors <- seq(0,1, length=100)
#myPalette <- colorRampPalette(c("red", "black", "green"))(n = 99)
gplots::heatmap.2(weight_mat_topK ,dendrogram = "none",Rowv = NA,Colv = NA,labRow = F ,srtCol = 0,trace = "none", density = "none", cellnote =gene_mat_topK, notecex = 0.5, notecol = "black", colsep=1:15,sepcolor="white",sepwidth=c(0.001,0.001), key.title = "Gene Probability", key.xlab ="Gene Probability",na.color = "black")
dev.off()

